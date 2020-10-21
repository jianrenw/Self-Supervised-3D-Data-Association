import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from scipy.stats import multivariate_normal
from .utils import points_in_bbox3D, bbox3D_to_corner, iou3d
import torch
import torch.nn as nn

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


class KalmanBoxTracker(object):
    """
      This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info, tc):
        """
    Initialises a tracker using initial bounding box.
    """
        # define constant velocity model
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

        self.kf.P[
            7:,
            7:] *= 1000.  #state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        self.kf.Q[7:, 7:] *= 0.01  # process uncertainty
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
        self.info = info
        self.tc = tc

    def update(self, bbox3D, info, tc):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the first time

        # orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta
        ) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        #########################

        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.info = info
        self.tc = tc

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

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

    def get_cov(self):
        """
        Returns the current bounding box covariance.
        """
        return self.kf.P[:7, :7]


def associate_detections_to_trackers(detections, trackers, matching_threshold,
                                     combine_alpha, mode, feature, confidence_mode):
    """
      Assigns detections to tracked object (both represented as bounding boxes)
      detections:
          corners: <np.float: N, 3, 8>
          points: <np.float: N, 3, npoint>
      trackers:
          corners: <np.float: M, 3, 8>
          points: <np.float: M, 3, npoint>
      Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    dets = detections['dets']['bbox3Ds']
    det_corners = detections['corners']
    det_points = detections['points']
    trks = trackers['trks']
    trk_corners = trackers['corners']
    trk_points = trackers['points']

    if (len(trks) == 0):
        return np.empty((0, 2), dtype=int), [], np.arange(
            len(det_corners)), np.empty((0), dtype=int)

    if mode == 'iou':
        iou_matrix = np.zeros((len(det_corners), len(trk_corners)),
                              dtype=np.float32)
        for d, det in enumerate(det_corners):
            for t, trk in enumerate(trk_corners):
                iou_matrix[d, t] = iou3d(det, trk)  # det: 3 x 8, trk: 3 x 8
        matching_matrix = iou_matrix
        prob_matrix = np.zeros((len(det_corners), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                prob_matrix[d, t] = multivariate_normal.pdf(
                    det, mean=trk.get_state(), cov=trk.get_cov())
    elif mode == 'prob':
        prob_matrix = np.zeros((len(det_corners), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                prob_matrix[d, t] = multivariate_normal.pdf(
                    det, mean=trk.get_state(), cov=trk.get_cov())
        matching_matrix = prob_matrix
    elif mode == 'ie':
        iou_matrix = np.zeros((len(det_corners), len(trk_corners)),
                              dtype=np.float32)
        for d, det in enumerate(det_corners):
            for t, trk in enumerate(trk_corners):
                iou_matrix[d, t] = iou3d(det, trk)  # det: 3 x 8, trk: 3 x 8
        if len(det_corners) != 0:
            with torch.no_grad():
                det_points_c = np.repeat(det_points, len(trk_corners), axis=0)
                trk_points_c = np.tile(trk_points, (len(det_corners), 1, 1))
                feature_d, _, _ = feature(
                    torch.from_numpy(det_points_c).cuda())
                feature_t, _, _ = feature(
                    torch.from_numpy(trk_points_c).cuda())
                embedding_similarity = cos(feature_d, feature_t).view(
                    len(det_corners), len(trk_corners)).detach().cpu().numpy()
                if combine_alpha is not None:
                    matching_matrix = iou_matrix * combine_alpha + (
                        1 + embedding_similarity) / 2 * (1 - combine_alpha)
                else:
                    matching_matrix = iou_matrix * (
                        1 + embedding_similarity) / 2
        else:
            matching_matrix = iou_matrix
    elif mode == 'pe':
        prob_matrix = np.zeros((len(det_corners), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                prob_matrix[d, t] = multivariate_normal.pdf(
                    det, mean=trk.get_state(), cov=trk.get_cov())
        if len(det_corners) != 0:
            with torch.no_grad():
                det_points_c = np.repeat(det_points, len(trk_corners), axis=0)
                trk_points_c = np.tile(trk_points, (len(det_corners), 1, 1))
                feature_d, _, _ = feature(
                    torch.from_numpy(det_points_c).cuda())
                feature_t, _, _ = feature(
                    torch.from_numpy(trk_points_c).cuda())
                embedding_similarity = cos(feature_d, feature_t).view(
                    len(det_corners), len(trk_corners)).detach().cpu().numpy()
                if combine_alpha is not None:
                    matching_matrix = prob_matrix * combine_alpha + (
                        1 + embedding_similarity) / 2 * (1 - combine_alpha)
                else:
                    matching_matrix = prob_matrix * (
                        1 + embedding_similarity) / 2
        else:
            matching_matrix = prob_matrix

    row_ind, col_ind = linear_sum_assignment(
        -matching_matrix)  # hungarian algorithm

    unmatched_detections = []
    for d, det in enumerate(det_corners):
        if (d not in row_ind):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trk_corners):
        if (t not in col_ind):
            unmatched_trackers.append(t)

    if confidence_mode == 'iou_joint':
        matches = []
        confidences = []
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        for m in matched_indices:
            if (matching_matrix[m[0], m[1]] < matching_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                s1 = matching_matrix[m[0], m[1]]
                try:
                    s2 = np.amax(np.delete(matching_matrix[m[0], :],
                                        m[1])) + 0.000000001
                except ValueError:
                    s2 = 0.000000001
                try:
                    s3 = np.amax(np.delete(matching_matrix[:, m[1]],
                                        m[0])) + 0.000000001
                except ValueError:
                    s3 = 0.000000001
                confidence = 1 - np.exp(-np.minimum(s1 / s2, s1 / s3))
                confidences.append(confidence)
                matches.append(m.reshape(1, 2))  # (detection, tracker)
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
    elif confidence_mode == 'iou_single':
        matches = []
        confidences = []
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        for m in matched_indices:
            if (matching_matrix[m[0], m[1]] < matching_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                s1 = matching_matrix[m[0], m[1]]
                confidence = s1
                confidences.append(confidence)
                matches.append(m.reshape(1, 2))  # (detection, tracker)
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
    elif confidence_mode == 'prob_joint':
        matches = []
        confidences = []
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        for m in matched_indices:
            if (matching_matrix[m[0], m[1]] < matching_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                s1 = prob_matrix[m[0], m[1]]
                try:
                    s2 = np.amax(np.delete(prob_matrix[m[0], :],
                                        m[1])) + 0.000000001
                except ValueError:
                    s2 = 0.000000001
                try:
                    s3 = np.amax(np.delete(prob_matrix[:, m[1]],
                                        m[0])) + 0.000000001
                except ValueError:
                    s3 = 0.000000001
                confidence = 1 - np.exp(-np.minimum(s1 / s2, s1 / s3))
                confidences.append(confidence)
                matches.append(m.reshape(1, 2))  # (detection, tracker)
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
    elif confidence_mode == 'prob_single':
        matches = []
        confidences = []
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        for m in matched_indices:
            if (matching_matrix[m[0], m[1]] < matching_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                s1 = prob_matrix[m[0], m[1]]
                confidence = s1
                confidences.append(confidence)
                matches.append(m.reshape(1, 2))  # (detection, tracker)
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

    return matches, confidences, np.array(unmatched_detections), np.array(
        unmatched_trackers)

class UA3DMOT(object):
    def __init__(self, max_age, min_hits, matching_threshold, combine_alpha,
                 mode, uncertainty_mode):
        self.max_age = max_age
        self.min_hits = min_hits
        self.matching_threshold = matching_threshold
        self.combine_alpha = combine_alpha
        self.mode = mode
        self.trackers = []
        self.frame_count = 0
        self.uncertainty_mode = uncertainty_mode

    def update(self, dets, points, feature):
        """
        Params:
            dets: dict
                bbox3Ds: <np.float: N, 7>
                infos: <np.float: N, 2> a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        bbox3Ds, infos = dets['bbox3Ds'], dets['infos']
        self.frame_count += 1

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

        dets_corner = [bbox3D_to_corner(bbox3D) for bbox3D in bbox3Ds]
        if len(dets_corner) > 0:
            dets_corner = np.stack(dets_corner, axis=0)
        trks_corner = [bbox3D_to_corner(trk_tmp) for trk_tmp in trks]
        if len(trks_corner) > 0:
            trks_corner = np.stack(trks_corner, axis=0)

        if feature is None:
            detections = {'dets': dets, 'corners': dets_corner, 'points': []}
            trackers = {
                'trks': self.trackers,
                'corners': trks_corner,
                'points': []
            }
        else:
            dets_point = [
                points_in_bbox3D(points, bbox3D) for bbox3D in bbox3Ds
            ]
            if len(dets_corner) > 0:
                dets_point = np.stack(dets_point, axis=0)
            trks_point = [
                points_in_bbox3D(points, trk_tmp) for trk_tmp in trks
            ]
            if len(trks_corner) > 0:
                trks_point = np.stack(trks_point, axis=0)

            detections = {
                'dets': dets,
                'corners': dets_corner,
                'points': dets_point
            }
            trackers = {
                'trks': self.trackers,
                'corners': trks_corner,
                'points': trks_point
            }
        matched, confidences, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trackers, self.matching_threshold, self.combine_alpha,
            self.mode, feature, self.uncertainty_mode)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0],
                            0]  # a list of index
                confidence = confidences[np.where(matched[:, 1] == t)[0][0]]
                trk.update(bbox3Ds[d, :][0], infos[d, :][0], confidence)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:  # a scalar of index
            trk = KalmanBoxTracker(bbox3Ds[i, :], infos[i, :], 1)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()  # bbox location

            if ((trk.time_since_update < self.max_age)
                    and (trk.hits >= self.min_hits
                         or self.frame_count <= self.min_hits)):
                ret.append(
                    np.concatenate(
                        (d, [trk.id + 1], trk.info, [trk.tc])).reshape(
                            1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update >= self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(
                ret
            )  # x, y, z, yaw, w, l, h, ID, class, confidence, tracking_confidence
        return np.empty((0, 10))
