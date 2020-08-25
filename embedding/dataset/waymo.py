# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import os
import numpy as np
import random
import torch.utils.data as data


class WAYMOGT(data.Dataset):
    def __init__(self, opt, mode):
        super(WAYMOGT, self).__init__()
        self.npoints = opt.npoints
        self.all_instances = []
        data_dir = os.path.join(opt.load_root, mode, opt.category)
        files = os.listdir(data_dir)
        for file in files:
            tmp_npz = np.load(os.path.join(data_dir, file))
            instance = [tmp_npz[key].astype(np.float32) for key in tmp_npz]
            self.all_instances.append(instance)

        self.instance_num = len(self.all_instances)

    def __getitem__(self, index):
        instance1 = self.all_instances[index]
        adversarial_index = list(range(self.instance_num))
        adversarial_index.remove(index)
        adversarial_index = random.choice(adversarial_index)
        instance2 = self.all_instances[adversarial_index]
        [point11, point12] = random.sample(instance1, 2)
        point2 = random.choice(instance2)
        if self.npoints <= point11.shape[1]:
            selected_point11 = point11[:3,
                                       np.random.choice(
                                           point11.shape[1],
                                           self.npoints,
                                           replace=False)]
        else:
            m = self.npoints // point11.shape[1]
            selected_point11 = point11[:3,
                                       np.concatenate(
                                           (np.tile(
                                               np.arange(point11.shape[1]), m),
                                            np.random.choice(
                                                point11.shape[1],
                                                self.npoints -
                                                point11.shape[1] * m,
                                                replace=False)))]
        if self.npoints <= point12.shape[1]:
            selected_point12 = point12[:3,
                                       np.random.choice(
                                           point12.shape[1],
                                           self.npoints,
                                           replace=False)]
        else:
            m = self.npoints // point12.shape[1]
            selected_point12 = point12[:3,
                                       np.concatenate(
                                           (np.tile(
                                               np.arange(point12.shape[1]), m),
                                            np.random.choice(
                                                point12.shape[1],
                                                self.npoints -
                                                point12.shape[1] * m,
                                                replace=False)))]
        if self.npoints <= point2.shape[1]:
            selected_point2 = point2[:3,
                                     np.random.choice(
                                         point2.shape[1],
                                         self.npoints,
                                         replace=False)]
        else:
            m = self.npoints // point2.shape[1]
            selected_point2 = point2[:3,
                                     np.concatenate(
                                         (np.
                                          tile(np.arange(point2.shape[1]), m),
                                          np.random.choice(
                                              point2.shape[1],
                                              self.npoints -
                                              point2.shape[1] * m,
                                              replace=False)))]
        return selected_point11, selected_point12, selected_point2

    def __len__(self):
        return self.instance_num
