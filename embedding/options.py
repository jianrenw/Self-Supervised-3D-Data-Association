# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # test settings

        # dataset settings
        self.parser.add_argument(
            '--num_workers',
            default=10,
            type=int,
            help='# threads for loading data')
        self.parser.add_argument(
            '--batch_size', type=int, default=64, help='batch size')

        # training settings

        self.parser.add_argument(
            '--data_parallel',
            type=int,
            default=0,
            help='use data parallel or not')
        self.parser.add_argument(
            '--dataset',
            type=str,
            default='NUSCENES',
            help='dataset to be used')
        self.parser.add_argument(
            '--mode', type=str, default='estss', help='training mode')
        self.parser.add_argument(
            '--load_root',
            type=str,
            default='/home/jianrenw/nuscenes_embedding',
            help='dataset root directory')
        self.parser.add_argument(
            '--confidence', default=1, type=int, help='use uncertainty or not')
        self.parser.add_argument(
            '--npoints', default=150, type=int, help='# points per instance')
        self.parser.add_argument(
            '--category', default='car', type=str, help='# training category')
        self.parser.add_argument(
            '--epoch', default=100, type=int, help='# total epoch')
        self.parser.add_argument(
            '--lr', type=float, default=0.00002, help='learning rate')
        self.parser.add_argument(
            '--beta1', type=float, default=0.9, help='learning rate')
        self.parser.add_argument(
            '--beta2', type=float, default=0.999, help='learning rate')
        self.parser.add_argument(
            '--log_dir',
            type=str,
            default='/home/jianrenw/logs/3dmot',
            help='log output directory')
        self.parser.add_argument(
            '--iter', default=70000, type=int, help='# total iterations')
        self.parser.add_argument(
            '--log_iter', type=int, default=100, help='log frequency')
        self.parser.add_argument(
            '--save_iter', type=int, default=1000, help='save frequency')
        self.parser.add_argument(
            '--experiment_name',
            type=str,
            default='nuscenes_estss',
            help='Current Experiment Name')
        self.parser.add_argument(
            '--resume_exp',
            type=str,
            default='nuscenes_estss',
            help='Resume Experiment Name')
        self.parser.add_argument(
            '--eval',
            type=int,
            default=1,
            help='Evaluate during training or not.')

        # PointNet settings

        self.parser.add_argument(
            '--feature_size',
            type=int,
            default=1024,
            help='pointnet global feature size')
        self.parser.add_argument(
            '--global_feat',
            default=1,
            type=int,
            help='whether the output is global feature')
        self.parser.add_argument(
            '--feature_transform',
            default=0,
            type=int,
            help='whether use feature transform function')

        # Visualize Setting

        self.parser.add_argument(
            '--ep', type=int, default=4, help='load model')
        self.parser.add_argument(
            '--visualize_save_dir',
            type=str,
            default='./visualize',
            help='visualize save directory')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
