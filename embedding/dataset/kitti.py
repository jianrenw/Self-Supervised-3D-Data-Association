from torch.utils.data import Dataset
from utils.data_classes import PointCloud, Box

from pyquaternion import Quaternion

import numpy as np
import pandas as pd
import os
import random

# For reproducibility
random.seed(0)
np.random.seed(0)

class kittiDataset():
    def __init__(self, noise, path):
        self.noise = noise
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [14]
            else:
                sceneID = list(range(0, 17))
        elif "VALID" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [3]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(10))
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib',
                                  anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box

    def getListOfAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path))
            and int(path) in sceneID
        ]
        list_of_tracklet_anno = []
        for scene in list_of_scene:
            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [
                    anno for index, anno in df_tracklet.iterrows()
                ]
                list_of_tracklet_anno.append(tracklet_anno)
        return list_of_tracklet_anno

    def getListOfAllAnno(self, sceneID, category_name="Car"):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path))
            and int(path) in sceneID
        ]
        list_of_all_anno = {}
        for scene in list_of_scene:
            list_of_all_anno[scene] = {}
            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df = df[df["type"] == category_name]
            df.insert(loc=0, column="scene", value=scene)
            for frame in df.frame.unique():
                df_tracklet = df[df["frame"] == frame]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [
                    anno for index, anno in df_tracklet.iterrows()
                ]
                list_of_all_anno[scene][frame] = tracklet_anno

        return list_of_all_anno

    def getBBIDs(self, boxs):
        BBs = []
        IDs = []
        for box in boxs:
            if self.noise:
                center = [
                    box["x"] + (random.random() * 0.2 - 0.1) * box["width"],
                    box["y"] - box["height"] / 2 +
                    +(random.random() * 0.2 - 0.1) * box["height"],
                    box["z"] + (random.random() * 0.2 - 0.1) * box["length"]
                ]
                size = [(random.random() * 0.2 + 0.9) * box["width"],
                        (random.random() * 0.2 + 0.9) * box["length"],
                        (random.random() * 0.2 + 0.9) * box["height"]]
                orientation = Quaternion(
                    axis=[0, 1, 0],
                    radians=box["rotation_y"] + random.random() * 0.0872 -
                    0.0436) * Quaternion(
                        axis=[1, 0, 0], radians=np.pi / 2)
            else:
                center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
                size = [box["width"], box["length"], box["height"]]
                orientation = Quaternion(
                    axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                        axis=[1, 0, 0], radians=np.pi / 2)
            BB = Box(center, size, orientation)
            BBs.append(BB)
            IDs.append(box['track_id'])
        return BBs, IDs

    def getPCandBBfromPandas(self, box, calib):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
                                         f'{box["frame"]:06}.bin')
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
            PC.transform(calib)
        except FileNotFoundError:
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0]]).T)

        return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data


class TripletDataset(Dataset):
    def __init__(self,
                 path,
                 split,
                 noise,
                 category_name="Car",
                 offset_BB=0,
                 scale_BB=1.0):

        self.dataset = kittiDataset(noise, path=path)

        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC
        self.getBBIDs = self.dataset.getBBIDs

        self.category_name = category_name

        self.list_of_tracklet_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name)
        self.list_of_all_anno = self.dataset.getListOfAllAnno(
            self.sceneID, category_name)

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class TripletTest(TripletDataset):
    def __init__(self,
                 path,
                 split="",
                 noise=0,
                 category_name="Car",
                 offset_BB=0,
                 scale_BB=1.0):
        super().__init__(
            path=path,
            split=split,
            noise=noise,
            category_name=category_name,
            offset_BB=offset_BB,
            scale_BB=scale_BB)
        self.split = split
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB

    def getitem(self, index):
        list_of_anno = self.list_of_tracklet_anno[index]
        PCs = []
        BBs = []
        All_BBs = []
        All_IDs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.getBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
            current_BBs, current_IDs = self.getBBIDs(
                self.list_of_all_anno[anno['scene']][anno['frame']])
            All_BBs.append(current_BBs)
            All_IDs.append(current_IDs)
        return PCs, BBs, All_BBs, All_IDs, list_of_anno

    def __len__(self):
        return len(self.list_of_tracklet_anno)
