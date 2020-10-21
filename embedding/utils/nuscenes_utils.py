import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud

category_list = [
    'construction_vehicle', 'motorcycle', 'car', 'bicycle', 'truck', 'barrier',
    'pedestrian', 'bus', 'trailer', 'traffic_cone'
]

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}


def quaternion_to_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def tsrcs_to_bbox3D(translation, size, rotation):
    """
    Returns bbox3D <np.float: 7, > [x, y, z, yaw, w, l, h].
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    yaw = quaternion_to_yaw(Quaternion(rotation))

    return np.array([
        translation[0], translation[1], translation[2], yaw, size[0], size[1],
        size[2]
    ])


def get_all_sample_tokens(nusc, name_token, scene_name):
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


def get_points_data(nusc, all_sample_tokens):
    all_points = []
    all_ego_poses = []
    for sample_token in all_sample_tokens:
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
        points[3, :] = points[3, :] / 255.0
        all_points.append(points)
        all_ego_poses.append(ps_record)
    return all_points, all_ego_poses


def get_gts(nusc, all_sample_tokens, scene_id):
    all_gts = []
    all_instances = {}
    for sample_id, sample_token in enumerate(all_sample_tokens):
        sample = nusc.get('sample', sample_token)
        anns_tokens = sample['anns']
        gts = []
        for anns_token in anns_tokens:
            annotation = nusc.get('sample_annotation', anns_token)
            if annotation['category_name'] in NameMapping:
                bbox3D = tsrcs_to_bbox3D(annotation['translation'],
                                         annotation['size'],
                                         annotation['rotation'])
                gts.append([
                    annotation['instance_token'],
                    category_list.index(
                        NameMapping[annotation['category_name']]), bbox3D
                ])
                if annotation['instance_token'] in all_instances:
                    all_instances[annotation['instance_token']].append([
                        category_list.index(
                            NameMapping[annotation['category_name']]),
                        scene_id, sample_id, bbox3D
                    ])
                else:
                    all_instances[annotation['instance_token']] = [[
                        category_list.index(
                            NameMapping[annotation['category_name']]),
                        scene_id, sample_id, bbox3D
                    ]]
        all_gts.append(gts)
    return all_gts, all_instances


def get_all_gts(mode):
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='/home/jianrenw/data/nuscenes/trainval',
                    verbose=True)
    if mode == 'train':
        scene_names = splits.train
    elif mode == 'val':
        scene_names = splits.val
    name_token = {}
    for scene in nusc.scene:
        name_token[scene['name']] = scene['token']
    all_all_instances = {}
    for scene_id, scene_name in enumerate(scene_names):
        all_sample_tokens = get_all_sample_tokens(nusc, name_token, scene_name)
        _, all_instances = get_gts(nusc, all_sample_tokens, scene_id)
        all_all_instances.update(all_instances)
    return all_all_instances
