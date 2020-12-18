# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import os.path as osp
from tqdm import tqdm

from trainer import Trainer
import random
from dataset.nuscenes_data import NUSCENES
from model.pointnet import PointNetfeat
from model.losses import TripletLoss
from utils.utils import get_dataloader, COLORS
from dataset.kitti import TripletTest
from utils.utils import cropAndCenterPC
import torch.nn.functional as F

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

class TripletTrainer(Trainer):
    def __init__(self, opt):
        super(TripletTrainer, self).__init__(opt)
        # take in all configurations
        self.opt = opt
        # set up dataloader
        self.train_dataloader, self.val_dataloader = self.setup_data()
        # Setup network components
        self.setup_nets()
        # Setup network losses
        self.setup_losses()
        # Setup optimizers
        self.setup_optimizers()

    def setup_data(self):
        """
        Set up dataloaders
        Return
        """
        train_dataset = NUSCENES(self.opt, 'train')
        train_dataloader = get_dataloader(self.opt, train_dataset, True)
        val_dataset = NUSCENES(self.opt, 'val')
        val_dataloader = get_dataloader(self.opt, val_dataset, False)
        return train_dataloader, val_dataloader

    def setup_nets(self):
        """
        Set up network components
        :return:
        """
        if not self.opt.data_parallel:
            feature = PointNetfeat(self.opt).cuda()
        else:
            feature = nn.DataParallel(PointNetfeat(self.opt).cuda())
        print(feature)
        setattr(self, "feature", feature)

    def setup_losses(self):
        triplet_loss = TripletLoss()
        setattr(self, "triplet_loss", triplet_loss)

    def setup_optimizers(self):
        optimizer = Adam(params=list(self.feature.parameters()),
                         lr=self.opt.lr,
                         betas=(self.opt.beta1, self.opt.beta2))
        setattr(self, "optimizer", optimizer)

    def train(self):
        save_dict = {}
        self.feature.train()

        ep_bar = tqdm(range(self.opt.epoch))
        for ep in ep_bar:
            batch_bar = tqdm(self.train_dataloader)
            for batch, data in enumerate(batch_bar):
                it = ep * len(self.train_dataloader) + batch
                ###############################################################
                # 0. Get the data
                ###############################################################
                if self.opt.hard_mining:
                    bases, positives, negative_candidates, confidences = data
                    bases, positives, negative_candidates, confidences = bases.cuda(
                    ), positives.cuda(), negative_candidates.cuda(
                    ), confidences.cuda()
                    negatives = torch.zeros(bases.size()[0], 3,
                                            self.opt.npoints).cuda()
                    for i, (negative_candidate,
                            base) in enumerate(zip(negative_candidates,
                                                   bases)):
                        base = base.unsqueeze(0).repeat((10, 1, 1))
                        negative_embedding, _, _ = self.feature(
                            negative_candidate)
                        base_embedding, _, _ = self.feature(base)
                        output = F.cosine_similarity(negative_embedding,
                                                     base_embedding,
                                                     dim=1)
                        scores = output.detach().cpu().numpy()
                        idx = np.argmax(scores)
                        negatives[i] = negative_candidate[idx]
                else:
                    bases, positives, negatives, confidences = data
                    bases, positives, negatives, confidences = bases.cuda(
                    ), positives.cuda(), negatives.cuda(), confidences.cuda()

                ###############################################################
                # 1. Get Features
                ###############################################################
                bases_features, _, _ = self.feature(bases)
                positives_features, _, _ = self.feature(positives)
                negatives_features, _, _ = self.feature(negatives)

                ###############################################################
                # 2. Compute triplet loss
                ###############################################################
                if self.opt.confidence:
                    loss, _, _ = self.triplet_loss(bases_features,
                                                   positives_features,
                                                   negatives_features,
                                                   confidences)
                else:
                    loss, _, _ = self.triplet_loss(bases_features,
                                                   positives_features,
                                                   negatives_features, None)

                # Aggregate the loss logs
                save_dict['loss'] = loss.item()

                if it % self.opt.log_iter == 0:
                    self.log('train', save_dict, ep, it, batch_bar)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_bar.close()
            if ep % self.opt.save_iter == 0:
                self.save_model(ep)
            if ep % self.opt.val_iter == 0:
                self.val(ep, it, batch_bar)
        ep_bar.close()

    def val(self, ep, it, pbar):
        save_dict = {}
        val_loss = 0
        val_accuracy = 0
        batch_num = 0
        dataloader = self.val_dataloader
        # calculate val loss
        with torch.no_grad():
            self.feature.eval()
            for data in dataloader:
                ###############################################################
                # 0. Get the data
                ###############################################################
                bases, positives, negatives, confidences = data
                bases, positives, negatives, confidences = bases.cuda(
                ), positives.cuda(), negatives.cuda(), confidences.cuda()

                ###############################################################
                # 1. Get Features
                ###############################################################
                bases_features, _, _ = self.feature(bases)
                positives_features, _, _ = self.feature(positives)
                negatives_features, _, _ = self.feature(negatives)

                ###############################################################
                # 2. Compute triplet loss
                ###############################################################
                if self.opt.confidence:
                    loss, positive_distance, negative_distance = self.triplet_loss(bases_features,
                                                   positives_features,
                                                   negatives_features,
                                                   confidences)
                else:
                    loss, positive_distance, negative_distance = self.triplet_loss(bases_features,
                                                   positives_features,
                                                   negatives_features, None)
                val_loss += loss.data

                ###############################################################
                # 3. Calculate accuracy
                ###############################################################
                pred = (positive_distance - negative_distance).data
                pred[pred >= 0] = 0
                pred[pred < 0] = 1
                val_accuracy += np.mean(pred.detach().cpu().numpy())
                batch_num += 1
        save_dict['loss'] = val_loss / batch_num
        save_dict['accuracy'] = val_accuracy / batch_num
        self.log('val', save_dict, ep, it, pbar)
        self.feature.train()

    def log(self, mode, save_dict, ep, it, pbar):
        print_str = "Epoch %03d, batch %05d:" % (ep, it)
        if mode == 'train':
            for var, val in save_dict.items():
                print_str += "%s %s %s %.4f," % (COLORS.WARNING, mode, var,
                                                 val)
                self.tb_train_writer.add_scalar(var, val, global_step=it)
        elif mode == 'val':
            for var, val in save_dict.items():
                print_str += "%s %s %s %.4f," % (COLORS.OKBLUE, mode, var, val)
                self.tb_val_writer.add_scalar(var, val, global_step=it)

        print_str += COLORS.ENDC
        pbar.write(print_str)

    def save_model(self, ep):
        # Save the optimizer's state_dict
        torch.save(self.optimizer.state_dict(),
                   osp.join(self.model_save_dir, "optimizer-%03d.ckpt" % ep))

        # Save the networks' state_dict
        torch.save(self.predictor.state_dict(),
                   osp.join(self.model_save_dir, "predictor-%03d.ckpt" % ep))

    def resume(self, ep):
        # Load the optimizer's state_dict
        self.optimizer.load_state_dict(
            torch.load(
                osp.join(self.model_save_dir, "optimizer-%03d.ckpt" % ep)))

        # Load the networks' state_dict
        self.predictor.load_state_dict(
            torch.load(
                osp.join(self.model_save_dir, "predictor-%03d.ckpt" % ep)))