# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

from abc import ABC, abstractmethod
import os
import tensorboardX


class Trainer(ABC):
    def __init__(self, opt):

        # States of the Trainer that is going to be shared across every subclasses
        self.lr = opt.lr
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2

        self.model_save_dir = "%s/%s/checkpoints/%s" % (
            opt.log_dir, opt.experiment_name, opt.category)
        self.tb_train_log_dir = "%s/tensorboard/%s/%s/%s" % (
            opt.log_dir,
            opt.experiment_name,
            opt.category,
            'train'
        )
        self.tb_val_log_dir = "%s/tensorboard/%s/%s/%s" % (
            opt.log_dir,
            opt.experiment_name,
            opt.category,
            'val'
        )

        self.model_resume_dir = "%s/%s/checkpoints/%s" % (
            opt.log_dir, opt.experiment_name, opt.category)
        self.create_log_dirs()

        self.tb_train_writer = tensorboardX.SummaryWriter(
            self.tb_train_log_dir)
        self.tb_val_writer = tensorboardX.SummaryWriter(self.tb_val_log_dir)

    def create_log_dirs(self):
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.tb_train_log_dir, exist_ok=True)
        os.makedirs(self.tb_val_log_dir, exist_ok=True)

    @abstractmethod
    def train(self):
        """Train method"""
        pass

    @abstractmethod
    def val(self):
        """Validation"""
        pass

    @abstractmethod
    def log(self, loss_dict, epoch, it, pbar):
        """Logger to print results and stuff"""
        pass

    @abstractmethod
    def setup_data(self):
        """Set up dataloaders for training/val/test"""
        pass

    @abstractmethod
    def setup_nets(self):
        """Set up network components for training/val/test"""
        pass

    @abstractmethod
    def setup_losses(self):
        """Set up network loss functions"""
        pass

    @abstractmethod
    def setup_optimizers(self):
        """Set up the network optimizers"""
        pass

    @abstractmethod
    def save_model(self, ep):
        """Save the model checkpoints"""
        pass

    @abstractmethod
    def resume(self, ep):
        """Resume from checkpoint"""
        pass
