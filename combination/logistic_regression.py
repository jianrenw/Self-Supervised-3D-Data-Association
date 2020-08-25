import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim import SGD
import tensorboardX
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='car')
parser.add_argument('--experiment', type=str, default='nuscenes_car_gt_2e-5')
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--mode', type=str, default='mda')
args = parser.parse_args()

os.makedirs(osp.join(
    '/home/jianrenw/logs/SGU/regression_{}_{}/weights'.format(
        args.mode, args.skip), args.experiment),
            exist_ok=True)

all_skip_positive = []
all_skip_negative = []

if args.skip:
    for i in range(1, args.skip):
        with open(
                osp.join(
                    '/data/all/jianrenw/SGU/nuscenes_track_mda_skip_{}'.format(
                        i), args.category, args.experiment,
                    'positive_logits.pkl'), 'rb') as f:
            positive_logits = pickle.load(f)
            all_skip_positive.append(positive_logits)

        with open(
                osp.join(
                    '/data/all/jianrenw/SGU/nuscenes_track_mda_skip_{}'.format(
                        i), args.category, args.experiment,
                    'negative_logits.pkl'), 'rb') as f:
            negative_logits = pickle.load(f)
            all_skip_negative.append(negative_logits)
else:
    with open(
            osp.join('/data/all/jianrenw/SGU/nuscenes_track_mda_skip_1',
                     args.category, args.experiment, 'positive_logits.pkl'),
            'rb') as f:
        positive_logits = pickle.load(f)
        all_skip_positive.append(positive_logits)
    with open(
            osp.join('/data/all/jianrenw/SGU/nuscenes_track_mda_skip_1',
                     args.category, args.experiment, 'negative_logits.pkl'),
            'rb') as f:
        negative_logits = pickle.load(f)
        all_skip_negative.append(negative_logits)


class Logistics(data.Dataset):
    def __init__(self, all_skip_positive, all_skip_negative, tv_mode, mode):
        super(Logistics, self).__init__()
        self.positive_points = []
        self.negative_points = []
        for skip_positive, skip_negative in zip(all_skip_positive,
                                                all_skip_negative):
            temp_positive_points = np.array(skip_positive)
            temp_negative_points = np.array(skip_negative)
            temp_selected_negative_points = temp_negative_points[
                np.random.choice(temp_negative_points.shape[0],
                                 temp_positive_points.shape[0],
                                 replace=False)]
            self.positive_points.append(temp_positive_points)
            self.negative_points.append(temp_selected_negative_points)
        self.positive_points = np.concatenate(self.positive_points, axis=0)
        self.negative_points = np.concatenate(self.negative_points, axis=0)
        if tv_mode == 'train':
            begin_p = 0
            end_p = int(self.positive_points.shape[0] * 0.7)
        elif tv_mode == 'test':
            begin_p = int(self.positive_points.shape[0] * 0.7 + 1)
            end_p = -1
        if mode == 'mda':
            self.positive_points = self.positive_points[begin_p:end_p]
            self.negative_points = self.negative_points[begin_p:end_p]
        else:
            self.positive_points = self.positive_points[begin_p:end_p][:,
                                                                       0:3:2]
            self.negative_points = self.negative_points[begin_p:end_p][:,
                                                                       0:3:2]
        self.positive_points[:, 0] = (self.positive_points[:, 0] + 1e-5) / 600
        self.positive_points[:, 1] = (self.positive_points[:, 1] + 1) / 2
        self.negative_points[:, 0] = (self.negative_points[:, 0] + 1e-5) / 600
        self.negative_points[:, 1] = (self.negative_points[:, 1] + 1) / 2
        self.log_positive_points = np.log(self.positive_points)
        self.log_negative_points = np.log(self.negative_points)

    def __len__(self):
        return self.positive_points.shape[0] * 2

    def __getitem__(self, index):
        if index % 2 == 0:
            index = int(index / 2)
            point = np.concatenate(
                [self.positive_points[index], self.log_positive_points[index]])
            label = np.ones(1)
        else:
            index = int((index - 1) / 2)
            point = np.concatenate(
                [self.negative_points[index], self.log_negative_points[index]])
            label = np.zeros(1)
        return point.astype(np.float32), label.astype(np.float32)


trainset = Logistics(all_skip_positive, all_skip_negative, 'train', args.mode)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=256,
                                          shuffle=True,
                                          num_workers=8)

testset = Logistics(all_skip_positive, all_skip_negative, 'test', args.mode)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=256,
                                         shuffle=False,
                                         num_workers=8)

if args.mode == 'mda':

    class regression(nn.Module):
        def __init__(self):
            super(regression, self).__init__()
            self.fc1 = torch.nn.Linear(6, 1, bias=True)

        def forward(self, x):
            return self.fc1(x)
else:

    class regression(nn.Module):
        def __init__(self):
            super(regression, self).__init__()
            self.fc1 = torch.nn.Linear(4, 1, bias=True)

        def forward(self, x):
            return self.fc1(x)


classifier = regression().cuda()
logit_loss = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = SGD(params=list(classifier.parameters()), lr=2e-3)
trian_writer = tensorboardX.SummaryWriter(
    osp.join(
        '/home/jianrenw/logs/SGU/regression_{}_{}/tensorboard'.format(
            args.mode, args.skip), 'train', args.experiment))
val_writer = tensorboardX.SummaryWriter(
    osp.join(
        '/home/jianrenw/logs/SGU/regression_{}_{}/tensorboard'.format(
            args.mode, args.skip), 'val', args.experiment))

epoch = 500

for i in range(epoch):
    train_loss = []
    val_loss = []
    classifier.train()
    for batch, train_sample in enumerate(trainloader):
        points, labels = train_sample
        points, labels = points.cuda(), labels.cuda()
        predictions = classifier(points)
        loss = logit_loss(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    classifier.eval()
    for batch, test_sample in enumerate(testloader):
        points, labels = test_sample
        points, labels = points.cuda(), labels.cuda()
        predictions = classifier(points)
        loss = logit_loss(predictions, labels)
        val_loss.append(loss.item())

    trian_writer.add_scalar('loss',
                            np.mean(np.array(train_loss)),
                            global_step=i)
    val_writer.add_scalar('loss', np.mean(np.array(val_loss)), global_step=i)
    if i % 50 == 0:
        torch.save(
            optimizer.state_dict(),
            osp.join(
                '/home/jianrenw/logs/SGU/regression_{}_{}/weights'.format(
                    args.mode, args.skip), args.experiment,
                'optimizer-%03d.ckpt' % i))
        torch.save(
            classifier.state_dict(),
            osp.join(
                '/home/jianrenw/logs/SGU/regression_{}_{}/weights'.format(
                    args.mode, args.skip), args.experiment,
                'classifier-%03d.ckpt' % i))
