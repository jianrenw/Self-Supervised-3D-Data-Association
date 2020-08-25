from nuscenes import NuScenes
import random
from filterpy.kalman import KalmanFilter
import numpy as np
from pyquaternion import Quaternion
from model.pointnet import PointNetfeat
from utils.utils import points_in_bbox3D, iou3d, bbox3D_to_corner
import os
import os.path as osp
import torch
import torch.nn.functional as F
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.stats import multivariate_normal
from nuscenes.utils import splits
import copy
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='car')
parser.add_argument('--experiment', type=str, default='nuscenes_car_gt_2e-5')
parser.add_argument('--skip_frame', type=int, default=2)
args = parser.parse_args()

save_root = osp.join(
    '/data/all/jianrenw/SGU/nuscenes_track_mda_skip_{}'.format(args.skip_frame),
    args.category, args.experiment)
if not osp.isdir(save_root):
    os.makedirs(save_root, exist_ok=True)

nusc = NuScenes(version='v1.0-trainval',
                dataroot='/data/all/jianrenw/nuscenes/trainval',
                verbose=True)
scene_names = splits.val
category_map = {
    'car': ['fd69059b62a3469fbaef25340c0eab7f'],
    'pedestrian': [
        '1fa93b757fc74fb197cdd60001ad8abf', 'b1c6de4c57f14a5383d9f963fbdcb5cb',
        'bb867e2064014279863c71a29b1eb381', '909f1237d34a49d6bdd27c2fe4581d79'
    ],
    'motorcycle': ['dfd26f200ade4d24b540184e16050022'],
    'bicycle': ['fc95c87b806f48f8a1faea2dcc2222a4'],
    'bus':
    ['003edbfb9ca849ee8a7496e9af3025d4', 'fedb11688db84088883945752e480c2c'],
    'truck': ['6021b5187b924d64be64a702e5570edf'],
    'trailer': ['90d0f6f8e7c749149b1b6c3a029841a8']
}
name_map = {
    'car': ['vehicle.car'],
    'pedestrian': [
        'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer'
    ],
    'motorcycle': ['vehicle.motorcycle'],
    'bicycle': ['vehicle.bicycle'],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'truck': ['vehicle.truck'],
    'trailer': ['vehicle.trailer']
}

name_token = {}
for scene in nusc.scene:
    name_token[scene['name']] = scene['token']
eval_tokens = []
for val_scene in scene_names[90:]:
    eval_tokens.append(name_token[val_scene])

eval_instances = []
for instance in nusc.instance:
    if instance['category_token'] in category_map[args.category] and nusc.get(
            'sample',
            nusc.get('sample_annotation', instance['first_annotation_token'])
        ['sample_token'])['scene_token'] in eval_tokens:
        eval_instances.append(instance)


class opt(object):
    def __init__(self):
        self.feature_size = 1024
        self.global_feat = 1
        self.feature_transform = 0


opt = opt()

model = PointNetfeat(opt).eval().cuda()
model_dir = osp.join('/home/jianrenw/logs/3dmot', args.experiment,
                     'checkpoints', args.category)
pretrained_dict = torch.load(
    osp.join(model_dir, 'feature-{}.ckpt'.format(70000)))
model.load_state_dict(pretrained_dict)


class Covariance(object):
    '''
  Define different Kalman Filter covariance matrix
  Kalman Filter states:
  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
  '''
    def __init__(self, covariance_id):
        if covariance_id == 2:
            self.num_states = 11  # with angular velocity
        else:
            self.num_states = 10
        self.num_observations = 7
        self.P = np.eye(self.num_states)
        self.Q = np.eye(self.num_states)
        self.R = np.eye(self.num_observations)

        NUSCENES_TRACKING_NAMES = [
            'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer',
            'truck'
        ]

        if covariance_id == 0:
            # default from baseline code
            self.P[self.num_observations:, self.num_observations:] *= 1000.
            self.P *= 10.
            self.Q[self.num_observations:, self.num_observations:] *= 0.01
        elif covariance_id == 1:
            # from kitti stats
            self.P[0, 0] = 0.01969623
            self.P[1, 1] = 0.01179107
            self.P[2, 2] = 0.04189842
            self.P[3, 3] = 0.52534431
            self.P[4, 4] = 0.11816206
            self.P[5, 5] = 0.00983173
            self.P[6, 6] = 0.01602004
            self.P[7, 7] = 0.01334779
            self.P[8, 8] = 0.00389245
            self.P[9, 9] = 0.01837525

            self.Q[0, 0] = 2.94827444e-03
            self.Q[1, 1] = 2.18784125e-03
            self.Q[2, 2] = 6.85044585e-03
            self.Q[3, 3] = 1.10964054e-01
            self.Q[4, 4] = 0
            self.Q[5, 5] = 0
            self.Q[6, 6] = 0
            self.Q[7, 7] = 2.94827444e-03
            self.Q[8, 8] = 2.18784125e-03
            self.Q[9, 9] = 6.85044585e-03

            self.R[0, 0] = 0.01969623
            self.R[1, 1] = 0.01179107
            self.R[2, 2] = 0.04189842
            self.R[3, 3] = 0.52534431
            self.R[4, 4] = 0.11816206
            self.R[5, 5] = 0.00983173
            self.R[6, 6] = 0.01602004

        elif covariance_id == 2:
            # nuscenes
            # see get_nuscenes_stats.py for the details on  how the numbers come from
            #Kalman Filter state: [x, y, z, rot_z, l, w, h, x_dot, y_dot, z_dot, rot_z_dot]

            P = {
                'bicycle': [
                    0.05390982, 0.05039431, 0.01863044, 1.29464435, 0.02713823,
                    0.01169572, 0.01295084, 0.04560422, 0.04097244, 0.01725477,
                    1.21635902
                ],
                'bus': [
                    0.17546469, 0.13818929, 0.05947248, 0.1979503, 0.78867322,
                    0.05507407, 0.06684149, 0.13263319, 0.11508148, 0.05033665,
                    0.22529652
                ],
                'car': [
                    0.08900372, 0.09412005, 0.03265469, 1.00535696, 0.10912802,
                    0.02359175, 0.02455134, 0.08120681, 0.08224643, 0.02266425,
                    0.99492726
                ],
                'motorcycle': [
                    0.04052819, 0.0398904, 0.01511711, 1.06442726, 0.03291016,
                    0.00957574, 0.0111605, 0.0437039, 0.04327734, 0.01465631,
                    1.30414345
                ],
                'pedestrian': [
                    0.03855275, 0.0377111, 0.02482115, 2.0751833, 0.02286483,
                    0.0136347, 0.0203149, 0.04237008, 0.04092393, 0.01482923,
                    2.0059979
                ],
                'trailer': [
                    0.23228021, 0.22229261, 0.07006275, 1.05163481, 1.37451601,
                    0.06354783, 0.10500918, 0.2138643, 0.19625241, 0.05231335,
                    0.97082174
                ],
                'truck': [
                    0.14862173, 0.1444596, 0.05417157, 0.73122169, 0.69387238,
                    0.05484365, 0.07748085, 0.10683797, 0.10248689, 0.0378078,
                    0.76188901
                ]
            }

            Q = {
                'bicycle': [
                    1.98881347e-02, 1.36552276e-02, 5.10175742e-03,
                    1.33430252e-01, 0, 0, 0, 1.98881347e-02, 1.36552276e-02,
                    5.10175742e-03, 1.33430252e-01
                ],
                'bus': [
                    1.17729925e-01, 8.84659079e-02, 1.17616440e-02,
                    2.09050032e-01, 0, 0, 0, 1.17729925e-01, 8.84659079e-02,
                    1.17616440e-02, 2.09050032e-01
                ],
                'car': [
                    1.58918523e-01, 1.24935318e-01, 5.35573165e-03,
                    9.22800791e-02, 0, 0, 0, 1.58918523e-01, 1.24935318e-01,
                    5.35573165e-03, 9.22800791e-02
                ],
                'motorcycle': [
                    3.23647590e-02, 3.86650974e-02, 5.47421635e-03,
                    2.34967407e-01, 0, 0, 0, 3.23647590e-02, 3.86650974e-02,
                    5.47421635e-03, 2.34967407e-01
                ],
                'pedestrian': [
                    3.34814566e-02, 2.47354921e-02, 5.94592529e-03,
                    4.24962535e-01, 0, 0, 0, 3.34814566e-02, 2.47354921e-02,
                    5.94592529e-03, 4.24962535e-01
                ],
                'trailer': [
                    4.19985099e-02, 3.68661552e-02, 1.19415050e-02,
                    5.63166240e-02, 0, 0, 0, 4.19985099e-02, 3.68661552e-02,
                    1.19415050e-02, 5.63166240e-02
                ],
                'truck': [
                    9.45275998e-02, 9.45620374e-02, 8.38061721e-03,
                    1.41680460e-01, 0, 0, 0, 9.45275998e-02, 9.45620374e-02,
                    8.38061721e-03, 1.41680460e-01
                ]
            }

            R = {
                'bicycle': [
                    0.05390982, 0.05039431, 0.01863044, 1.29464435, 0.02713823,
                    0.01169572, 0.01295084
                ],
                'bus': [
                    0.17546469, 0.13818929, 0.05947248, 0.1979503, 0.78867322,
                    0.05507407, 0.06684149
                ],
                'car': [
                    0.08900372, 0.09412005, 0.03265469, 1.00535696, 0.10912802,
                    0.02359175, 0.02455134
                ],
                'motorcycle': [
                    0.04052819, 0.0398904, 0.01511711, 1.06442726, 0.03291016,
                    0.00957574, 0.0111605
                ],
                'pedestrian': [
                    0.03855275, 0.0377111, 0.02482115, 2.0751833, 0.02286483,
                    0.0136347, 0.0203149
                ],
                'trailer': [
                    0.23228021, 0.22229261, 0.07006275, 1.05163481, 1.37451601,
                    0.06354783, 0.10500918
                ],
                'truck': [
                    0.14862173, 0.1444596, 0.05417157, 0.73122169, 0.69387238,
                    0.05484365, 0.07748085
                ]
            }

            self.P = {
                tracking_name: np.diag(P[tracking_name])
                for tracking_name in NUSCENES_TRACKING_NAMES
            }
            self.Q = {
                tracking_name: np.diag(Q[tracking_name])
                for tracking_name in NUSCENES_TRACKING_NAMES
            }
            self.R = {
                tracking_name: np.diag(R[tracking_name])
                for tracking_name in NUSCENES_TRACKING_NAMES
            }
        else:
            assert (False)


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self,
                 bbox3D,
                 info,
                 track_confidence=1,
                 covariance_id=2,
                 track_score=1,
                 tracking_name='car',
                 use_angular_velocity=True):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        if not use_angular_velocity:
            self.kf = KalmanFilter(dim_x=10, dim_z=7)
            self.kf.F = np.array([
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ])

            self.kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            ])
        else:
            # with angular velocity
            self.kf = KalmanFilter(dim_x=11, dim_z=7)
            self.kf.F = np.array([
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ])

            self.kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            ])

        # Initialize the covariance matrix, see covariance.py for more details
        if covariance_id == 0:  # exactly the same as AB3DMOT baseline
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            self.kf.P[
                7:,
                7:] *= 1000.  #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.

            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            self.kf.Q[7:, 7:] *= 0.01
        elif covariance_id == 1:  # for kitti car, not supported
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P
            self.kf.Q = covariance.Q
            self.kf.R = covariance.R
        elif covariance_id == 2:  # for nuscenes
            covariance = Covariance(covariance_id)
            self.kf.P = covariance.P[tracking_name]
            self.kf.Q = covariance.Q[tracking_name]
            self.kf.R = covariance.R[tracking_name]
            if not use_angular_velocity:
                self.kf.P = self.kf.P[:-1, :-1]
                self.kf.Q = self.kf.Q[:-1, :-1]
        else:
            assert (False)

        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info
        self.track_confidence = track_confidence
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity

    def update(self, bbox3D, info, track_confidence):
        """ 
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        ######################### orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta
        ) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        self.info = info
        self.track_confidence = track_confidence

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:7].reshape((7, ))


def angle_in_range(angle):
    '''
    Input angle: -2pi ~ 2pi
    Output angle: -pi ~ pi
    '''
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle


def diff_orientation_correction(det, trk):
    '''
    return the angle diff = det - trk
    if angle diff > 90 or < -90, rotate trk and update the angle diff
    '''
    diff = det - trk
    diff = angle_in_range(diff)
    if diff > np.pi / 2:
        diff -= np.pi
    if diff < -np.pi / 2:
        diff += np.pi
        diff = angle_in_range(diff)
    return diff


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
    points[3, :] = points[3, :] / 255.0
    return points[:3, :]


def track_n_frame(skip_frame, tracker, previous_embedding, current_token,
                  last_token):
    temp_positive_mda = []  # motion_det_appearance
    temp_negative_mda = []  # motion_det_appearance
    temp_tracker = copy.deepcopy(tracker)

    for i in range(skip_frame):
        pos = temp_tracker.predict()
    i = 1
    while current_token != last_token and i < skip_frame:
        gt_ann = nusc.get('sample_annotation', current_token)
        current_token = gt_ann['next']
        i += 1
    if i != skip_frame:
        return None, None
    trk_S = np.matmul(np.matmul(temp_tracker.kf.H, temp_tracker.kf.P),
                      temp_tracker.kf.H.T) + temp_tracker.kf.R
    S_inv = np.linalg.inv(trk_S)
    prediction = np.array(
        [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]])[:, 0]
    # get ground truth
    gt_ann = nusc.get('sample_annotation', current_token)
    q = Quaternion(gt_ann['rotation'])
    angle = q.angle if q.axis[2] > 0 else -q.angle
    gt_bb = np.array([
        gt_ann['translation'][0], gt_ann['translation'][1],
        gt_ann['translation'][2], angle, gt_ann['size'][0], gt_ann['size'][1],
        gt_ann['size'][2]
    ])
    gt_corner = bbox3D_to_corner(gt_bb)
    candidate_anns = data['results'][gt_ann['sample_token']]
    positive_bbs = []
    positive_dets = []
    positive_ious = []
    positive_scores = []
    negative_bbs = []
    negative_dets = []
    negative_ious = []
    negative_scores = []
    for candidate_ann in candidate_anns:
        if candidate_ann['detection_name'] != args.category:
            continue
        q = Quaternion(candidate_ann['rotation'])
        angle = q.angle if q.axis[2] > 0 else -q.angle
        candidate_det = np.array([
            candidate_ann['translation'][0], candidate_ann['translation'][1],
            candidate_ann['translation'][2], angle, candidate_ann['size'][1],
            candidate_ann['size'][0], candidate_ann['size'][2]
        ])
        candidate_bb = np.array([
            candidate_ann['translation'][0], candidate_ann['translation'][1],
            candidate_ann['translation'][2], angle, candidate_ann['size'][0],
            candidate_ann['size'][1], candidate_ann['size'][2]
        ])
        candidate_corner = bbox3D_to_corner(candidate_bb)
        iou = iou3d(gt_corner, candidate_corner)
        if iou > 0.6:  # Following PointPillar
            positive_bbs.append(candidate_bb)
            positive_dets.append(candidate_det)
            positive_ious.append(iou)
            positive_scores.append(candidate_ann['detection_score'])
        elif iou < 0.45:
            negative_bbs.append(candidate_bb)
            negative_dets.append(candidate_det)
            negative_ious.append(iou)
            negative_scores.append(candidate_ann['detection_score'])
    current_pc = get_points_data(gt_ann['sample_token'])
    for i in range(len(positive_ious)):
        diff = np.expand_dims(positive_dets[i] - prediction, axis=1)
        corrected_angle_diff = diff_orientation_correction(
            positive_dets[i][3], prediction[3])
        diff[3] = corrected_angle_diff
        positive_motion_dis = np.sqrt(
                np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
        tmp_embedding, _, _ = model(
            torch.from_numpy(
                points_in_bbox3D(current_pc, positive_bbs[i],
                                 150)[None, :, :]).cuda())
        output = F.cosine_similarity(previous_embedding, tmp_embedding, dim=1)
        positive_appear_similarity = output.detach().cpu().numpy()[0]
        temp_positive_mda.append([
            positive_motion_dis, positive_appear_similarity,
            positive_scores[i]
        ])
    for i in range(len(negative_ious)):
        diff = np.expand_dims(negative_dets[i] - prediction, axis=1)
        corrected_angle_diff = diff_orientation_correction(
            negative_dets[i][3], prediction[3])
        diff[3] = corrected_angle_diff
        negative_motion_dis = np.sqrt(
                        np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
        tmp_embedding, _, _ = model(
            torch.from_numpy(
                points_in_bbox3D(current_pc, negative_bbs[i],
                                 150)[None, :, :]).cuda())
        output = F.cosine_similarity(previous_embedding, tmp_embedding, dim=1)
        negative_appear_similarity = output.detach().cpu().numpy()[0]
        temp_negative_mda.append([
            negative_motion_dis, negative_appear_similarity,
            negative_scores[i]
        ])
    return temp_positive_mda, temp_negative_mda


# detection_file = '/data/all/jianrenw/nuscenes/detection/results_val.json'
detection_file = '/data/all/jianrenw/nuscenes/detection/infos_val_10sweeps_withvelo.json'

with open(detection_file) as f:
    data = json.load(f)
results = data['results']

positive_mda = []  # motion_det_appearance
negative_mda = []  # motion_det_appearance
for my_instance in eval_instances:
    first_token = my_instance['first_annotation_token']
    last_token = my_instance['last_annotation_token']
    current_token = first_token
    while current_token != last_token:
        gt_ann = nusc.get('sample_annotation', current_token)
        q = Quaternion(gt_ann['rotation'])
        angle = q.angle if q.axis[2] > 0 else -q.angle
        gt_bb = np.array([
            gt_ann['translation'][0], gt_ann['translation'][1],
            gt_ann['translation'][2], angle, gt_ann['size'][0],
            gt_ann['size'][1], gt_ann['size'][2]
        ])
        gt_corner = bbox3D_to_corner(gt_bb)
        candidate_anns = data['results'][gt_ann['sample_token']]
        positive_bbs = []
        positive_dets = []
        positive_ious = []
        positive_scores = []
        negative_bbs = []
        negative_dets = []
        negative_ious = []
        negative_scores = []
        for candidate_ann in candidate_anns:
            if candidate_ann['detection_name'] != args.category:
                continue
            q = Quaternion(candidate_ann['rotation'])
            angle = q.angle if q.axis[2] > 0 else -q.angle
            candidate_det = np.array([
                candidate_ann['translation'][0],
                candidate_ann['translation'][1],
                candidate_ann['translation'][2], angle,
                candidate_ann['size'][1], candidate_ann['size'][0],
                candidate_ann['size'][2]
            ])
            candidate_bb = np.array([
                candidate_ann['translation'][0],
                candidate_ann['translation'][1],
                candidate_ann['translation'][2], angle,
                candidate_ann['size'][0], candidate_ann['size'][1],
                candidate_ann['size'][2]
            ])
            candidate_corner = bbox3D_to_corner(candidate_bb)
            iou = iou3d(gt_corner, candidate_corner)
            if iou > 0.6:  # Following PointPillar
                positive_bbs.append(candidate_bb)
                positive_dets.append(candidate_det)
                positive_ious.append(iou)
                positive_scores.append(candidate_ann['detection_score'])
            elif iou < 0.45:
                negative_bbs.append(candidate_bb)
                negative_dets.append(candidate_det)
                negative_ious.append(iou)
                negative_scores.append(candidate_ann['detection_score'])
        current_pc = get_points_data(gt_ann['sample_token'])
        if len(
                positive_ious
        ) == 0:  # cannot be matched, use gt to update, but don't save the logits
            matched_detection = np.array([
                gt_ann['translation'][0], gt_ann['translation'][1],
                gt_ann['translation'][2], angle, gt_ann['size'][1],
                gt_ann['size'][0], gt_ann['size'][2]
            ])
            matched_embedding, _, _ = model(
                torch.from_numpy(
                    points_in_bbox3D(current_pc, gt_bb,
                                     150)[None, :, :]).cuda())
        else:
            matched_index = np.argmax(np.array(positive_ious))
            matched_detection = positive_dets[matched_index]
            # get points
            matched_bb = positive_bbs[matched_index]
            matched_embedding, _, _ = model(
                torch.from_numpy(
                    points_in_bbox3D(current_pc, matched_bb,
                                     150)[None, :, :]).cuda())
        if current_token == first_token:
            tracker = KalmanBoxTracker(matched_detection, np.array([1.0]))
        else:
            temp_positive_mda, temp_negative_mda = track_n_frame(
                args.skip_frame, tracker, previous_embedding, current_token,
                last_token)
            if temp_positive_mda is not None and temp_negative_mda is not None:
                positive_mda += temp_positive_mda
                negative_mda += temp_negative_mda
            tracker.update(matched_detection, np.array([1.0]), 1)
        previous_embedding = matched_embedding
        current_token = gt_ann['next']

with open(osp.join(save_root, 'positive_logits.pkl'), "wb") as fp:
    pickle.dump(positive_mda, fp)
with open(osp.join(save_root, 'negative_logits.pkl'), "wb") as fp:
    pickle.dump(negative_mda, fp)