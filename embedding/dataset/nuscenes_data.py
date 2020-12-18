# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import os
import os.path as osp
import numpy as np
import random
import torch.utils.data as data
import copy
import json
import pickle

# For reproducibility
random.seed(0)
np.random.seed(0)

def regularized_points(points, npoints):
    if npoints <= points.shape[1]:
        points = points[:3,
                        np.random.
                        choice(points.shape[1], npoints, replace=False)]
    else:
        m = npoints // points.shape[1]
        points = points[:3,
                        np.concatenate(
                            (np.tile(np.arange(points.shape[1]), m),
                             np.random.choice(points.shape[1],
                                              npoints - points.shape[1] * m,
                                              replace=False)))]
    return points


class NUSCENES(object):
    def __init__(self, opt, split):
        super(NUSCENES, self).__init__()
        with open(osp.join(opt.load_root, split, opt.mode, 'points.pkl'), 'rb') as f:
            all_points = pickle.load(f)
        self.objs = {}
        self.points = all_points[opt.category]
        self.npoints = opt.npoints
        self.hard = opt.hard_mining
        for sample_token, instances in self.points.items():
            for obj_id, info in instances.items():
                if obj_id in self.objs:
                    self.objs[obj_id].append([sample_token, info[1]])
                else:
                    self.objs[obj_id] = [[sample_token, info[1]]
        illegal_objs = []
        for obj_id, infos in self.objs:
            if len(infos) < 2:
                illegal_objs.append(obj_id)
        for illegal_obj in illegal_objs:
            self.objs.pop(illegal_obj)
        self.obj_ids = list(self.objs.keys())

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, index):
        confidence = 1.0
        instance1 = self.objs[self.obj_ids[index]]
        samples = np.sort(np.random.choice(len(instance1), 2))
        for k in range(samples[0], samples[1]):
            confidence *= instance1[k + 1][1]
        base_sample_token = self.objs[self.obj_ids[index]][
            samples[0]][0]
        positive_sample_token = self.objs[self.obj_ids[index]][
            samples[1]][0]
        base = regularized_points(
            self.points[base_sample_token][self.obj_ids[index]][0],
            self.npoints)
        positive = regularized_points(
            self.points[positive_sample_token][self.obj_ids[index]][0],
            self.npoints)
        if self.hard: # random select 10 negative samples
            counter = 0
            negative_candidates = []
            for obj_id, info in self.points[base_sample_token].items():
                if obj_id != self.obj_ids[index] and counter <= 9:
                    negative_candidates.append(
                        regularized_points(info[0], self.npoints))
                    counter += 1
            for _ in range(10 - len(negative_candidates)):
                negative_candidates.append(np.zeros((3, self.npoints)))
            negative_candidates = np.stack(negative_candidates)
            return base.astype(np.float32), positive.astype(
                np.float32), negative_candidates.astype(
                    np.float32), np.float32(confidence)
        else:
            adversarial_ids = list(self.points[base_sample_token].keys())
            adversarial_ids.remove(self.obj_ids[index])
            adversarial_id = random.choice(adversarial_ids)
            negative = regularized_points(
                self.points[base_sample_token][
                    adversarial_id][0], self.npoints)
            return base.astype(np.float32), positive.astype(
                np.float32), negative.astype(
                    np.float32), np.float32(confidence)
