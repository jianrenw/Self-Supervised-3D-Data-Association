# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import os
import os.path as osp
import numpy as np
import argparse
import json
import pickle
from nuscenes.utils import splits
from utils.utils import points_in_bbox3D
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

parser = argparse.ArgumentParser()
parser.add_argument('--result_path',
                    type=str,
                    default='/home/jianrenw/nuscenes_track')
parser.add_argument('--save_path',
                    type=str,
                    default='/home/jianrenw/nuscenes_embedding')
parser.add_argument('--split', type=str, default='train')  # train / val
parser.add_argument('--mode', type=str, default='estss')  # gtgt / gtss / estss

args = parser.parse_args()
with open(
        osp.join(args.result_path, args.split, args.mode,
                 'results_train_probabilistic_tracking.json')) as f:
    data = json.load(f)
    results = data['results']

# tracks threshold
# For metrics details, please run nuscenes evaluation code first
with open(osp.join(args.result_path, args.split, 'metrics_details.json')) as f:
    data = json.load(f)
thresholds = {}
for key, value in data.items():
    mota = np.array(value['mota'])
    mota = mota[~np.isnan(mota)]
    threshold = np.array(value['confidence'])
    threshold = threshold[~np.isnan(threshold)][np.argmax(mota)]
    thresholds[key] = threshold

nusc = NuScenes(version='v1.0-trainval',
                dataroot='/home/jianrenw/data/nuscenes/trainval',
                verbose=True)

if args.split == 'train':
    scene_names = splits.train
elif args.split == 'val':
    scene_names = splits.val

name_token = {}
for scene in nusc.scene:
    name_token[scene['name']] = scene['token']

category_token = [
    'fd69059b62a3469fbaef25340c0eab7f', '1fa93b757fc74fb197cdd60001ad8abf',
    'b1c6de4c57f14a5383d9f963fbdcb5cb', 'bb867e2064014279863c71a29b1eb381',
    '909f1237d34a49d6bdd27c2fe4581d79', 'dfd26f200ade4d24b540184e16050022',
    'fc95c87b806f48f8a1faea2dcc2222a4', '003edbfb9ca849ee8a7496e9af3025d4',
    'fedb11688db84088883945752e480c2c', '6021b5187b924d64be64a702e5570edf',
    '90d0f6f8e7c749149b1b6c3a029841a8'
]

name_map = {
    'vehicle.car': 'car',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.trailer': 'trailer'
}

category_list = ['car','pedestrian','motorcycle','bicycle','bus','truck','trailer']

def get_all_sample_tokens(name_token, scene_name):
    all_sample_tokens = []
    scene_token = name_token[scene_name]
    my_scene = nusc.get('scene', scene_token)
    nbr_samples = my_scene['nbr_samples']
    sample_token = my_scene['first_sample_token']
    all_sample_tokens.append(sample_token)
    my_sample = nusc.get('sample', my_scene['first_sample_token'])
    while my_sample['next'] != '':
        all_sample_tokens.append(my_sample['next'])
        my_sample = nusc.get('sample', my_sample['next'])
    assert nbr_samples == len(all_sample_tokens)
    return all_sample_tokens


def get_points_data(sample_token):
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    chan = 'LIDAR_TOP'
    ref_chan = 'LIDAR_TOP'
    pc, _ = LidarPointCloud.from_file_multisweep(nusc,
                                                 sample,
                                                 chan,
                                                 ref_chan,
                                                 nsweeps=10)
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         lidar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    ps_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pc.rotate(Quaternion(ps_record['rotation']).rotation_matrix)
    pc.translate(np.array(ps_record['translation']))
    points = pc.points
    return points[:3, :]


if args.mode == 'estss' or args.mode == 'gtss':
    tracks = {}
    for scene_name in scene_names:
        sample_tokens = get_all_sample_tokens(name_token, scene_name)
        for i, sample_token in enumerate(sample_tokens):
            objects = results[sample_token]
            for obj in objects:
                if obj['tracking_id'] in tracks:
                    tracks[obj['tracking_id']]['detection_scores'].append(
                        obj['tracking_score'])
                    tracks[obj['tracking_id']]['frame_id'].append(i)
                else:
                    tracks[obj['tracking_id']] = {
                        'detection_scores': [obj['tracking_score']],
                        'frame_id': [i],
                        'tracking_name': obj['tracking_name']
                    }

    # filter illegal tracks
    illegal_tracks = []
    for tracking_id, info in tracks.items():
        tracking_score = info['detection_scores']
        if len(tracking_score) < 2 or np.mean(
                np.array(tracking_score)
        ) < thresholds[info[
                'tracking_name']]:  # remove len of trajectory <= 2 or low score trajectories
            illegal_tracks.append(tracking_id)
    for illegal_track in illegal_tracks:
        tracks.pop(key)

    points = {}
    # save points of instances
    for cat in category_list:
        points[cat] = {}
        for scene_name in scene_names:
            sample_tokens = get_all_sample_tokens(name_token, scene_name)
            for i, sample_token in enumerate(sample_tokens):
                objects = results[sample_token]
                lidar_points = get_points_data(sample_token)
                temp = {}
                instance_num = 0
                for obj in objects:
                    if obj['tracking_name'] != cat:
                        continue
                    if obj['tracking_id'] in tracks:
                        q = Quaternion(obj['rotation'])
                        angle = q.angle if q.axis[2] > 0 else -q.angle
                        obj_bb = np.array([
                            obj['translation'][0], obj['translation'][1],
                            obj['translation'][2], angle, obj['size'][0],
                            obj['size'][1], obj['size'][2]
                        ])
                        instance_points = points_in_bbox3D(lidar_points, obj_bb,
                                                        None)
                        if instance_points.shape[1] < 20:
                            continue
                        instance_num += 1
                        temp[obj['tracking_id']] = [instance_points, obj['tracking_confidence']]
                if instance_num > 1:
                    points[cat][sample_token] = {}
                    for tracking_id, info in temp.items():
                        points[cat][sample_token][obj['tracking_id']] = info

    save_dir = osp.join(args.save_path, args.split, args.mode)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, 'points.pkl'), "wb") as fp:
        pickle.dump(points, fp)

elif args.mode == 'gtgt':
    legal_instances = []
    for instance in nusc.instance:
        if instance['category_token'] in category_token and instance[
                'nbr_annotations'] > 1:
            legal_instances.append(instance['token'])
    points = {}
    # save points of instances
    for cat in category_list:
        points[cat] = {}
        for scene_name in scene_names:
            sample_tokens = get_all_sample_tokens(name_token, scene_name)
            for i, sample_token in enumerate(sample_tokens):
                lidar_points = get_points_data(sample_token)
                temp = {}
                instance_num = 0
                sample_info = nusc.get('sample', sample_token)
                anno_tokens = sample_info['anns']
                for anno_token in anno_tokens:
                    gt_ann = nusc.get('sample_annotation', anno_token)
                    if name_map[gt_ann['category_name']] != cat:
                        continue
                    if gt_ann['instance_token'] in legal_instances:
                        q = Quaternion(gt_ann['rotation'])
                        angle = q.angle
                        gt_bb = np.array([
                            gt_ann['translation'][0], gt_ann['translation'][1],
                            gt_ann['translation'][2], angle, gt_ann['size'][0],
                            gt_ann['size'][1], gt_ann['size'][2]
                        ])
                        instance_points = points_in_bbox3D(lidar_points, gt_bb,
                                                        None)
                        if instance_points.shape[1] < 20:
                            continue
                        instance_num += 1
                        temp[gt_ann['instance_token']] = [instance_points, 1] # points, confidence
                if instance_num > 1: 
                    points[cat][sample_token] = {}
                    for tracking_id, info in temp.items():
                        points[cat][sample_token][
                                gt_ann['instance_token']] = info
    save_dir = osp.join(args.save_path, args.split, args.mode)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, 'points.pkl'), "wb") as fp:
        pickle.dump(points, fp)
