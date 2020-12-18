# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

from __future__ import print_function
import sys
import os.path, copy, numpy as np, time, sys
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
from covariance import Covariance
from scipy.stats import multivariate_normal
import json
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from tqdm import tqdm
sys.path.append('../embedding/model')
from pointnet import PointNetfeat, regression_mda, regression_md
sys.path.append('../embedding/utils')
from utils import points_in_bbox3D
from nuscenes_utils import tsrcs_to_bbox3D
import torch
import torch.nn.functional as F


def get_points_data(nusc, sample_token):
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


@jit
def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


@jit
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :])**2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :])**2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :])**2))
    return a * b * c


@jit
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] -
                                                      cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


@jit
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@jit
def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def convert_3dbox_to_8corner(bbox3d_input, nuscenes_to_kitti=False):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
        Note: the output of this function will be passed to the funciton iou3d
            for calculating the 3D-IOU. But the function iou3d was written for 
            kitti, so the caller needs to set nuscenes_to_kitti to True if 
            the input bbox3d_input is in nuscenes format.
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    if nuscenes_to_kitti:
        # transform to kitti format first
        bbox3d_nuscenes = copy.copy(bbox3d)
        # kitti:    [x,  y,  z,  a, l, w, h]
        # nuscenes: [y, -z, -x, -a, w, l, h]
        bbox3d[0] = bbox3d_nuscenes[1]
        bbox3d[1] = -bbox3d_nuscenes[2]
        bbox3d[2] = -bbox3d_nuscenes[0]
        bbox3d[3] = -bbox3d_nuscenes[3]
        bbox3d[4] = bbox3d_nuscenes[5]
        bbox3d[5] = bbox3d_nuscenes[4]

    R = roty(bbox3d[3])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]

    return np.transpose(corners_3d)


class KalmanBoxTracker(object):
    """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
    count = 0

    def __init__(self,
                 bbox3D,
                 info,
                 track_confidence,
                 mda_score,
                 covariance_id=0,
                 track_score=None,
                 tracking_name='car',
                 use_angular_velocity=False):
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
            print(covariance_id)
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
        self.mda_score = mda_score
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity

    def update(self, bbox3D, info, track_confidence, mda_score):
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
        self.mda_score = mda_score

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


def greedy_match(distance_matrix):
    '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks],
                        axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[
                tracking_id] == -1 and detection_id_matches_to_tracking_id[
                    detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices


def associate_detections_to_trackers(se,
                                     ass_mode,
                                     info,
                                     tracking_name,
                                     detections,
                                     trackers,
                                     current_points,
                                     previous_points,
                                     iou_threshold=0.1,
                                     use_mahalanobis=False,
                                     dets=None,
                                     trks=None,
                                     trks_S=None,
                                     mahalanobis_threshold=0.1,
                                     print_debug=False,
                                     match_algorithm='greedy',
                                     confidence_mode='alone'):
    """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  dets: N x 7
  trks: M x 7
  trks_S: M x 7 x 7

  Returns 4 lists of matches, confidences, unmatched_detections and unmatched_trackers
  """
    if (len(trackers) == 0):
        return np.empty(
            (0, 2), dtype=int), [], [], np.arange(len(detections)), np.empty(
                (0, 8, 3), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    distance_matrix = np.zeros((len(detections), len(trackers)),
                               dtype=np.float32)
    prob_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    len_det = (len(detections))
    len_track = (len(trackers))

    if use_mahalanobis:
        assert (dets is not None)
        assert (trks is not None)
        assert (trks_S is not None)

    if use_mahalanobis and print_debug:
        print('dets.shape: ', dets.shape)
        print('dets: ', dets)
        print('trks.shape: ', trks.shape)
        print('trks: ', trks)
        print('trks_S.shape: ', trks_S.shape)
        print('trks_S: ', trks_S)
        S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
        S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]  # 7
        print('S_inv_diag: ', S_inv_diag)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if use_mahalanobis:
                S_inv = np.linalg.inv(trks_S[t])  # 7 x 7
                diff = np.expand_dims(dets[d] - trks[t], axis=1)  # 7 x 1
                # manual reversed angle by 180 when diff > 90 or < -90 degree
                corrected_angle_diff = diff_orientation_correction(
                    dets[d][3], trks[t][3])
                diff[3] = corrected_angle_diff
                distance_matrix[d, t] = np.sqrt(
                    np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])
            else:
                iou_matrix[d, t] = iou3d(det, trk)[0]  # det: 8 x 3, trk: 8 x 3
                distance_matrix = -iou_matrix
            if confidence_mode == 'alone':
                prob_matrix[d, t] = multivariate_normal.pdf(dets[d],
                                                            mean=trks[t],
                                                            cov=trks_S[t])

    if se is not None:
        cal_dis = distance_matrix.reshape((-1, 1))
        cal_dis = torch.from_numpy(cal_dis).cuda()
        cal_det = np.array(info).astype(np.float32)
        cal_det = np.repeat(cal_det, len_track, axis=0)
        cal_det = torch.from_numpy(cal_det).cuda()
        if ass_mode == 'mda':
            if previous_points is not None and current_points is not None:
                appearance_similarity = se.get_appearance_similarity(
                    tracking_name, previous_points, current_points, len_det,
                    len_track)
                score = se.get_score(tracking_name, 'mda', cal_dis,
                                     appearance_similarity, cal_det, len_det,
                                     len_track)
            else:
                score = se.get_score(tracking_name, 'md', cal_dis, None,
                                     cal_det, len_det, len_track)
        else:
            score = se.get_score(tracking_name, 'md', cal_dis, None, cal_det,
                                 len_det, len_track)
    else:
        score = distance_matrix
        # S_det = np.linalg.det(trks_S[t])
        # prob_matrix[d, t] = 1 / np.sqrt((2*np.pi)**7*S_det) * np.exp(-1/2*np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])

    if match_algorithm == 'greedy':
        if se is None:
            matched_indices = greedy_match(score)
        else:
            matched_indices = greedy_match(-score)
    elif match_algorithm == 'pre_threshold':
        if use_mahalanobis:
            to_max_mask = distance_matrix > mahalanobis_threshold
            distance_matrix[to_max_mask] = mahalanobis_threshold + 1
        else:
            to_max_mask = iou_matrix < iou_threshold
            distance_matrix[to_max_mask] = 0
            iou_matrix[to_max_mask] = 0
        matched_indices = linear_assignment(
            distance_matrix)  # houngarian algorithm
    else:
        matched_indices = linear_assignment(
            distance_matrix)  # houngarian algorithm

    if print_debug:
        print('distance_matrix.shape: ', distance_matrix.shape)
        print('distance_matrix: ', distance_matrix)
        print('matched_indices: ', matched_indices)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if len(matched_indices) == 0 or (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU and get confidence
    matches = []
    confidences = []
    mda_scores = []
    for m in matched_indices:
        match = True
        if use_mahalanobis:
            if distance_matrix[m[0], m[1]] > mahalanobis_threshold:
                match = False
        else:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                match = False
        if not match:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            mda_scores.append(score[m[0], m[1]])
            if use_mahalanobis:
                if confidence_mode == 'alone':
                    confidences.append(prob_matrix[m[0], m[1]])
                elif confidence_mode == 'joint':
                    confidence_matrix = np.power(distance_matrix, 1 / 3)
                    s1 = confidence_matrix[m[0], m[1]] + 1.0e-4
                    try:
                        s2 = np.amin(
                            np.delete(confidence_matrix[m[0], :], m[1]))
                    except ValueError:
                        s2 = 500
                    try:
                        s3 = np.amin(
                            np.delete(confidence_matrix[:, m[1]], m[0]))
                    except ValueError:
                        s3 = 500
                    confidences.append(1 -
                                       np.exp(-np.minimum(s2 / s1, s3 / s1)))
            else:
                if confidence_mode == 'alone':
                    confidences.append(iou_matrix[m[0], m[1]])
                elif confidence_mode == 'joint':
                    s1 = iou_matrix[m[0], m[1]]
                    try:
                        s2 = np.amax(np.delete(iou_matrix[m[0], :],
                                               m[1])) + 1.0e-10
                    except ValueError:
                        s2 = 1.0e-10
                    try:
                        s3 = np.amax(np.delete(iou_matrix[:, m[1]],
                                               m[0])) + 1.0e-10
                    except ValueError:
                        s3 = 1.0e-10
                    confidences.append(1 -
                                       np.exp(-np.minimum(s1 / s2, s1 / s3)))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    if print_debug:
        print('matches: ', matches)
        print('unmatched_detections: ', unmatched_detections)
        print('unmatched_trackers: ', unmatched_trackers)
    return matches, confidences, mda_scores, np.array(
        unmatched_detections), np.array(unmatched_trackers)


class AB3DMOT(object):
    def __init__(self,
                 covariance_id=0,
                 max_age=2,
                 min_hits=3,
                 tracking_name='car',
                 use_angular_velocity=False,
                 tracking_nuscenes=False):
        """              
    observation: 
      before reorder: [h, w, l, x, y, z, rot_y]
      after reorder:  [x, y, z, rot_y, l, w, h]
    state:
      [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
    """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.covariance_id = covariance_id
        self.tracking_name = tracking_name
        self.use_angular_velocity = use_angular_velocity
        self.tracking_nuscenes = tracking_nuscenes

    def update(self, se, tracking_name, dets_all, previous_lidar,
               current_lidar, match_distance, match_threshold, match_algorithm,
               confidence_mode, seq_name, ass_mode, show_hidden):
        """
        Params:
        dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets, info = dets_all['dets'], dets_all[
            'info']  # dets: N x 7, float numpy array
        #print('dets.shape: ', dets.shape)
        #print('info.shape: ', info.shape)
        dets = dets[:, self.reorder]

        self.frame_count += 1

        print_debug = False
        if False and seq_name == '2f56eb47c64f43df8902d9f88aa8a019' and self.frame_count >= 25 and self.frame_count <= 30:
            print_debug = True
            print('self.frame_count: ', self.frame_count)
        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        if previous_lidar is not None and current_lidar is not None:
            previous_points = [
                points_in_bbox3D(previous_lidar,
                                 tracker.get_state()[[0, 1, 2, 3, 5, 4, 6]],
                                 150) for tracker in self.trackers
            ]
            current_points = [
                points_in_bbox3D(current_lidar, det_tmp[[0, 1, 2, 3, 5, 4, 6]],
                                 150) for det_tmp in dets
            ]
        else:
            previous_points = None
            current_points = None

        trks = np.zeros(
            (len(self.trackers),
             7))  # N x 7 , #get predicted locations from existing trackers.
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict().reshape((-1, 1))
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if print_debug:
            for trk_tmp in self.trackers:
                print('trk_tmp.id: ', trk_tmp.id)

        dets_8corner = [
            convert_3dbox_to_8corner(
                det_tmp, match_distance == 'iou' and self.tracking_nuscenes)
            for det_tmp in dets
        ]
        if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
        else: dets_8corner = []

        trks_8corner = [
            convert_3dbox_to_8corner(
                trk_tmp, match_distance == 'iou' and self.tracking_nuscenes)
            for trk_tmp in trks
        ]
        trks_S = [
            np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) +
            tracker.kf.R for tracker in self.trackers
        ]
        if len(trks_8corner) > 0:
            trks_8corner = np.stack(trks_8corner, axis=0)
            trks_S = np.stack(trks_S, axis=0)
        if match_distance == 'iou':
            matched, confidences, mda_scores, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                se,
                ass_mode,
                info,
                tracking_name,
                dets_8corner,
                trks_8corner,
                current_points,
                previous_points,
                iou_threshold=match_threshold,
                print_debug=print_debug,
                match_algorithm=match_algorithm,
                confidence_mode=confidence_mode)
        else:
            matched, confidences, mda_scores, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
                se,
                ass_mode,
                info,
                tracking_name,
                dets_8corner,
                trks_8corner,
                current_points,
                previous_points,
                use_mahalanobis=True,
                dets=dets,
                trks=trks,
                trks_S=trks_S,
                mahalanobis_threshold=match_threshold,
                print_debug=print_debug,
                match_algorithm=match_algorithm,
                confidence_mode=confidence_mode)

        #update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0],
                            0]  # a list of index
                confidence = confidences[np.where(matched[:, 1] == t)[0][0]]
                mda_score = mda_scores[np.where(matched[:, 1] == t)[0][0]]
                trk.update(dets[d, :][0], info[d, :][0], confidence, mda_score)
                detection_score = info[d, :][0][-1]
                trk.track_score = detection_score

        #create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            detection_score = info[i][-1]
            track_score = detection_score
            trk = KalmanBoxTracker(dets[i, :], info[i, :], 1, 1,
                                   self.covariance_id, track_score,
                                   self.tracking_name, use_angular_velocity)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location
            d = d[self.reorder_back]

            if ((trk.time_since_update < self.max_age)
                    and (trk.hits >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                if show_hidden:
                    ret.append(
                        np.concatenate(
                            (d, [trk.id + 1], trk.info[:-1], [trk.track_score],
                             [trk.track_confidence], [trk.mda_score])).reshape(
                                 1,
                                 -1))  # +1 as MOT benchmark requires positive
                elif trk.time_since_update == 0:
                    ret.append(
                        np.concatenate(
                            (d, [trk.id + 1], trk.info[:-1], [trk.track_score],
                             [trk.track_confidence], [trk.mda_score])).reshape(
                                 1,
                                 -1))  # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(
                ret
            )  # x, y, z, theta, l, w, h, ID, other info(None), det confidence, track confidence, mda_score
        return np.empty((0, 11))


NUSCENES_TRACKING_NAMES = [
    'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck'
]

NUSCENES_NAME_MAPPING = {
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}


def format_sample_result(sample_token, tracking_name, tracker):
    '''
    Input:
        tracker: (11): [h, w, l, x, y, z, rot_y], tracking_id, tracking_score, track confidence, mda_score
    Output:
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "tracking_id":    <str>         -- Unique object id that is used to identify an object track across samples.
        "tracking_name":  <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                                        Note that the tracking_name cannot change throughout a track.
        "tracking_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                        We average over frame level scores to compute the track level score.
                                        The score is used to determine positive and negative tracks via thresholding.
        "tracking_confidence": <float>  -- Tracking confidence for association.
        "mda_score": <float>  -- logistic association score
    }
    '''
    rotation = Quaternion(axis=[0, 0, 1], angle=tracker[6]).elements
    sample_result = {
        'sample_token': sample_token,
        'translation': [tracker[3], tracker[4], tracker[5]],
        'size': [tracker[1], tracker[2], tracker[0]],
        'rotation': [rotation[0], rotation[1], rotation[2], rotation[3]],
        'velocity': [0, 0],
        'tracking_id': str(int(tracker[7])),
        'tracking_name': tracking_name,
        'tracking_score': tracker[8],
        'tracking_confidence': tracker[9],
        'mda_score': tracker[10]
    }

    return sample_result


class opt(object):
    def __init__(self):
        self.feature_size = 1024
        self.global_feat = 1
        self.feature_transform = 0


my_opt = opt()


class score_estimater(object):
    def __init__(self, embedding_mode, opt, NUSCENES_TRACKING_NAMES, use_skip,
                 max_age, detector_name):
        self.embedding_mode = embedding_mode
        self.NUSCENES_TRACKING_NAMES = NUSCENES_TRACKING_NAMES
        self.init_nns(opt)
        self.use_skip = use_skip
        self.max_age = max_age
        self.detector_name = detector_name

    def init_nns(self, opt):
        with torch.no_grad():
            self.feature = PointNetfeat(opt).eval().cuda()
            self.regression_mda = regression_mda().eval().cuda()
            self.regression_md = regression_md().eval().cuda()

    def get_appearance_similarity(self, category_name, previous_points,
                                  current_points, len_det, len_track):
        with torch.no_grad():
            if self.embedding_mode == 'gt':
                head = 'nuscenes'
            else:
                head = 'nuscenesuh_pro'
            model_dir = os.path.join(
                '/home/jianrenw/logs/3dmot',
                '{}_{}_{}_2e-5'.format(head, category_name,
                                       self.embedding_mode), 'checkpoints',
                category_name)
            pretrained_dict = torch.load(
                os.path.join(model_dir, 'feature-{}.ckpt'.format(70000)))
            self.feature.load_state_dict(pretrained_dict)
            try:
                previous_points = np.array(previous_points)
                current_points = np.array(current_points)
                cal_pre = np.tile(previous_points,
                                  (current_points.shape[0], 1, 1))
                cal_cur = np.repeat(current_points,
                                    previous_points.shape[0],
                                    axis=0)
                feature_1, _, _ = self.feature(
                    torch.from_numpy(cal_pre).cuda())
                feature_2, _, _ = self.feature(
                    torch.from_numpy(cal_cur).cuda())
                appearance_similarity = F.cosine_similarity(feature_1,
                                                            feature_2,
                                                            dim=1)
            except RuntimeError:
                appearance_similarity = torch.zeros(len_det, len_track).cuda()
                for i, previous_point in enumerate(previous_points):
                    temp_previous_points = np.array([previous_point])
                    current_points = np.array(current_points)
                    cal_pre = np.tile(temp_previous_points,
                                      (current_points.shape[0], 1, 1))
                    cal_cur = current_points
                    feature_1, _, _ = self.feature(
                        torch.from_numpy(cal_pre).cuda())
                    feature_2, _, _ = self.feature(
                        torch.from_numpy(cal_cur).cuda())
                    temp_appearance_similarity = F.cosine_similarity(feature_1,
                                                                     feature_2,
                                                                     dim=1)
                appearance_similarity[:, i] = temp_appearance_similarity
        return appearance_similarity.view(-1, 1)

    def get_score(self, category_name, association_mode, motion_dis,
                  appearance_similarity, detection_score, len_det, len_track):
        with torch.no_grad():
            if self.embedding_mode == 'gt':
                head = 'nuscenes'
            else:
                head = 'nuscenesuh_pro'
            if association_mode == 'mda':
                if self.use_skip:
                    model_dir = os.path.join(
                        '/home/jianrenw/logs/{}/regression_{}_{}/weights'.
                        format(self.detector_name, 'mda', self.max_age),
                        '{}_{}_{}_2e-5'.format(head, category_name,
                                               self.embedding_mode))
                else:
                    model_dir = os.path.join(
                        '/home/jianrenw/logs/{}/regression_{}_{}/weights'.
                        format(self.detector_name, 'mda', 1),
                        '{}_{}_{}_2e-5'.format(head, category_name,
                                               self.embedding_mode))
                pretrained_dict = torch.load(
                    os.path.join(model_dir, 'classifier-{}.ckpt'.format(450)))
                self.regression_mda.load_state_dict(pretrained_dict)
                motion_dis = (motion_dis + 1e-5) / 600
                appearance_similarity = (appearance_similarity + 1) / 2
                point = torch.cat(
                    (motion_dis, appearance_similarity, detection_score,
                     torch.log(motion_dis), torch.log(appearance_similarity),
                     torch.log(detection_score)),
                    axis=1)
                prediction = self.regression_mda(point)
                score = torch.sigmoid(prediction).detach().cpu().numpy()
                score = score.reshape((len_det, len_track))
            else:
                model_dir = os.path.join(
                    '/home/jianrenw/logs/{}/regression_{}_{}/weights'.format(
                        self.detector_name, 'md', 0),
                    '{}_{}_{}_2e-5'.format(head, category_name,
                                           self.embedding_mode))
                pretrained_dict = torch.load(
                    os.path.join(model_dir, 'classifier-{}.ckpt'.format(450)))
                self.regression_md.load_state_dict(pretrained_dict)
                motion_dis = (motion_dis + 1e-5) / 600
                point = torch.cat(
                    (motion_dis, detection_score, torch.log(motion_dis),
                     torch.log(detection_score)),
                    axis=1)
                prediction = self.regression_md(point)
                score = torch.sigmoid(prediction).detach().cpu().numpy()
                score = score.reshape((len_det, len_track))
        return score


def track_nuscenes(data_split, covariance_id, match_distance, match_threshold,
                   match_algorithm, confidence_mode, save_root,
                   use_angular_velocity, use_gt, use_mda, embedding_mode,
                   max_age, ass_mode, use_skip, detector_name, show_hidden):
    '''
  submission {
    "meta": {
        "use_camera":   <bool>  -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>  -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>  -- Whether this submission uses radar data as an input.
        "use_map":      <bool>  -- Whether this submission uses map data as an input.
        "use_external": <bool>  -- Whether this submission uses external data as an input.
    },
    "results": {
        sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
  }
  
  '''
    save_dir = os.path.join(save_root, data_split, str(max_age))
    mkdir_if_missing(save_dir)
    if 'train' in data_split:
        detection_file = '/volume00/data/brick/jianrenw/nuscenes/detection/results_train.json'
        data_root = '/volume00/data/brick/jianrenw/nuscenes/trainval'
        version = 'v1.0-trainval'
        output_path = os.path.join(
            save_dir, 'results_train_probabilistic_tracking.json')
    elif 'val' in data_split:
        if detector_name == 'megvii':
            detection_file = '/volume00/data/brick/jianrenw/nuscenes/detection/results_val.json'
        elif detector_name == 'SGU':
            detection_file = '/volume00/data/brick/jianrenw/nuscenes/detection/infos_val_10sweeps_withvelo.json'
        elif detector_name == 'SGU03':
            detection_file = '/home/jianrenw/data/det_03.json'
        elif detector_name == 'SGU04':
            detection_file = '/home/jianrenw/data/det_04.json'
        data_root = '/volume00/data/brick/jianrenw/nuscenes/trainval'
        version = 'v1.0-trainval'
        output_path = os.path.join(save_dir,
                                   'results_val_probabilistic_tracking.json')
    elif 'test' in data_split:
        detection_file = '/volume00/data/brick/jianrenw/nuscenes/detection/results_test.json'
        data_root = '/volume00/data/brick/jianrenw/nuscenes/test'
        version = 'v1.0-test'
        output_path = os.path.join(
            save_dir, 'results_test_probabilistic_tracking_{}.json'.format(
                embedding_mode))

    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    if use_mda:
        se = score_estimater(embedding_mode, my_opt, NUSCENES_TRACKING_NAMES,
                             use_skip, max_age, detector_name)
    else:
        se = None
    results = {}

    total_time = 0.0
    total_frames = 0

    with open(detection_file) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
      'See https://www.nuscenes.org/object-detection for more information.'

    all_results = EvalBoxes.deserialize(data['results'], DetectionBox)
    meta = data['meta']
    print('meta: ', meta)
    print("Loaded results from {}. Found detections for {} samples.".format(
        detection_file, len(all_results.sample_tokens)))

    processed_scene_tokens = set()
    for sample_token_idx in tqdm(range(len(all_results.sample_tokens))):
        sample_token = all_results.sample_tokens[sample_token_idx]
        scene_token = nusc.get('sample', sample_token)['scene_token']
        if scene_token in processed_scene_tokens:
            continue
        first_sample_token = nusc.get('scene',
                                      scene_token)['first_sample_token']
        current_sample_token = first_sample_token

        mot_trackers = {
            tracking_name: AB3DMOT(covariance_id,
                                   max_age=max_age,
                                   tracking_name=tracking_name,
                                   use_angular_velocity=use_angular_velocity,
                                   tracking_nuscenes=True)
            for tracking_name in NUSCENES_TRACKING_NAMES
        }
        previous_lidar = None
        while current_sample_token != '':
            results[current_sample_token] = []
            dets = {
                tracking_name: []
                for tracking_name in NUSCENES_TRACKING_NAMES
            }
            info = {
                tracking_name: []
                for tracking_name in NUSCENES_TRACKING_NAMES
            }
            if use_mda and ass_mode == 'mda':
                current_lidar = get_points_data(nusc, current_sample_token)
            else:
                current_lidar = None
            if use_gt:
                frame = nusc.get('sample', current_sample_token)
                anns_tokens = frame['anns']
                for anns_token in anns_tokens:
                    annotation = nusc.get('sample_annotation', anns_token)
                    if annotation[
                            'category_name'] not in NUSCENES_NAME_MAPPING:
                        continue
                    q = Quaternion(annotation['rotation'])
                    angle = q.angle if q.axis[2] > 0 else -q.angle
                    detection = np.array([
                        annotation['size'][2], annotation['size'][0],
                        annotation['size'][1], annotation['translation'][0],
                        annotation['translation'][1],
                        annotation['translation'][2], angle
                    ])
                    #print('detection: ', detection)
                    information = np.array([1.0])
                    dets[NUSCENES_NAME_MAPPING[
                        annotation['category_name']]].append(detection)
                    info[NUSCENES_NAME_MAPPING[
                        annotation['category_name']]].append(information)
            else:
                for box in all_results.boxes[current_sample_token]:
                    if box.detection_name not in NUSCENES_TRACKING_NAMES:
                        continue
                    q = Quaternion(box.rotation)
                    angle = q.angle if q.axis[2] > 0 else -q.angle
                    #print('box.rotation,  angle, axis: ', box.rotation, q.angle, q.axis)
                    #print('box.rotation,  angle, axis: ', q.angle, q.axis)
                    #[h, w, l, x, y, z, rot_y]
                    detection = np.array([
                        box.size[2], box.size[0], box.size[1],
                        box.translation[0], box.translation[1],
                        box.translation[2], angle
                    ])
                    #print('detection: ', detection)
                    information = np.array([box.detection_score])
                    dets[box.detection_name].append(detection)
                    info[box.detection_name].append(information)

            dets_all = {
                tracking_name: {
                    'dets': np.array(dets[tracking_name]),
                    'info': np.array(info[tracking_name])
                }
                for tracking_name in NUSCENES_TRACKING_NAMES
            }

            total_frames += 1
            start_time = time.time()
            for tracking_name in NUSCENES_TRACKING_NAMES:
                if dets_all[tracking_name]['dets'].shape[0] > 0:
                    trackers = mot_trackers[tracking_name].update(
                        se, tracking_name, dets_all[tracking_name],
                        previous_lidar, current_lidar, match_distance,
                        match_threshold, match_algorithm, confidence_mode,
                        scene_token, ass_mode, show_hidden)
                    # (N, 10)
                    # (h, w, l, x, y, z, rot_y), tracking_id, detection_score, tracking_confidence
                    # print('trackers: ', trackers)
                    for i in range(trackers.shape[0]):
                        sample_result = format_sample_result(
                            current_sample_token, tracking_name, trackers[i])
                        results[current_sample_token].append(sample_result)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # get next frame and continue the while loop
            current_sample_token = nusc.get('sample',
                                            current_sample_token)['next']
            previous_lidar = current_lidar
        # left while loop and mark this scene as processed
        processed_scene_tokens.add(scene_token)

    # finished tracking all scenes, write output data
    output_data = {'meta': meta, 'results': results}
    with open(output_path, 'w') as outfile:
        json.dump(output_data, outfile)

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))


if __name__ == '__main__':
    if len(sys.argv) != 18:
        print(len(sys.argv))
        print(
            "Usage: python main.py data_split(train, val, test) covariance_id(0, 1, 2) match_distance(iou or m) match_threshold match_algorithm(greedy or h) use_angular_velocity(true or false) dataset save_root"
        )
        sys.exit(1)

    data_split = sys.argv[1]
    covariance_id = int(sys.argv[2])
    match_distance = sys.argv[3]
    match_threshold = float(sys.argv[4])
    match_algorithm = sys.argv[5]
    use_angular_velocity = sys.argv[6] == 'True' or sys.argv[6] == 'true'
    dataset = sys.argv[7]
    save_root = sys.argv[8]
    confidence_mode = sys.argv[9]
    use_gt = sys.argv[10] == 'True' or sys.argv[10] == 'true'
    use_mda = sys.argv[11] == 'True' or sys.argv[11] == 'true'
    embedding_mode = sys.argv[12]
    max_age = int(sys.argv[13])  # 2,3,4,5,6
    ass_mode = sys.argv[14]
    use_skip = sys.argv[15] == 'True' or sys.argv[15] == 'true'
    detector_name = sys.argv[16]
    show_hidden = sys.argv[17] == 'True' or sys.argv[17] == 'true'

    if dataset == 'kitti':
        print('track kitti not supported')
    elif dataset == 'nuscenes':
        print('track nuscenes')
        track_nuscenes(data_split, covariance_id, match_distance,
                       match_threshold, match_algorithm, confidence_mode,
                       save_root, use_angular_velocity, use_gt, use_mda,
                       embedding_mode, max_age, ass_mode, use_skip,
                       detector_name, show_hidden)
