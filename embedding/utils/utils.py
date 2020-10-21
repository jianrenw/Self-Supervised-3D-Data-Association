# Author: Jianren Wang
# email: jianrenwang.cs@gmail.com

import numpy as np
import collections
from scipy.spatial import ConvexHull
from torch.utils.data import DataLoader
from pyquaternion import Quaternion
import copy
from utils.data_classes import PointCloud


def bbox3D_to_corner(bbox3D):
    """
    Returns bbox3D <np.float: 7, > [x, y, z, yaw, w, l, h].
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    x, y, z = bbox3D[:3]
    w, l, h = bbox3D[4:]
    yaw = bbox3D[3]
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(rotation_matrix, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def points_in_bbox3D(points, bbox3D, npoints=150, scale=1.1):
    """
    Checks whether points are inside the box.
    :param points: <np.float: 3, n>.
    :param bbox3D <np.float: 7, > [x, y, z, yaw, w, l, h].
    :return: <np.float: 3, p>.
    """
    x, y, z = bbox3D[:3]
    w, l, h = bbox3D[4:]
    yaw = -bbox3D[3]
    r = np.sqrt((l / 2)**2 + (w / 2)**2)
    candidate = np.stack((points[0, :] > x - 2 * r, points[0, :] < x + 2 * r,
                          points[1, :] > y - 2 * r, points[1, :] < y + 2 * r,
                          points[2, :] > z - h, points[2, :] < z + h))
    candidate_points_idx = np.nonzero(np.all(candidate, axis=0))[0]
    candidate_points = points[:, candidate_points_idx] - np.array(
        [x, y, z]).reshape(-1, 1)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    candidate_points = np.matmul(rotation, candidate_points)
    idx = ((candidate_points[0, :] >= -l / 2 * scale) &
           (candidate_points[0, :] <= l / 2 * scale) &
           (candidate_points[1, :] >= -w / 2 * scale) &
           (candidate_points[1, :] <= w / 2 * scale) &
           (candidate_points[2, :] >= -h / 2 * scale) &
           (candidate_points[2, :] <= h / 2 * scale))
    obj_points = candidate_points[:, idx]
    if npoints is None:
        return obj_points[:3, :].astype('float32')
    if npoints <= obj_points.shape[1]:
        selected_points = obj_points[:3,
                                     np.random.choice(
                                         obj_points.shape[1],
                                         npoints,
                                         replace=False)]
    else:
        if obj_points.shape[1] == 0:
            selected_points = np.zeros((3, npoints))
        else:
            m = npoints // obj_points.shape[1]
            selected_points = obj_points[:3,
                                         np.concatenate(
                                             (np.tile(
                                                 np.arange(obj_points.
                                                           shape[1]), m),
                                              np.random.choice(
                                                  obj_points.shape[1],
                                                  npoints -
                                                  obj_points.shape[1] * m,
                                                  replace=False)))]
    return selected_points.astype('float32')


def box3d_vol(corners):
    ''' corners: (3,8) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[:, 2] - corners[:, 3])**2))
    b = np.sqrt(np.sum((corners[:, 2] - corners[:, 6])**2))
    c = np.sqrt(np.sum((corners[:, 2] - corners[:, 1])**2))
    return a * b * c


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
    if collections.Counter(subjectPolygon) == collections.Counter(clipPolygon):
        return subjectPolygon

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (
            p[0] - cp1[0])

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


def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (3,8), ^Z [2376][1045]
        corners2: numpy array (3,8), ^Z
    Output:
        iou: 3D bounding box IoU
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[0, i], corners1[1, i]) for i in [2, 3, 7, 6]]
    rect2 = [(corners2[0, i], corners2[1, i]) for i in [2, 3, 7, 6]]
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    zmax = min(corners1[2, 0], corners2[2, 0])
    zmin = max(corners1[2, 3], corners2[2, 3])
    inter_vol = inter_area * max(0.0, zmax - zmin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return np.clip(iou, 0, 0.99999999999)


def calculate_distances(bbox3D1s, bbox3D2s, max_iou):
    gts_corner = [bbox3D_to_corner(bbox3D) for bbox3D in bbox3D1s]
    hts_corner = [bbox3D_to_corner(bbox3D) for bbox3D in bbox3D2s]
    gt_len = len(gts_corner)
    ht_len = len(hts_corner)
    if gt_len == 0 or ht_len == 0:
        return []
    distances_matrix = np.ones((gt_len, ht_len))
    for i in range(gt_len):
        for j in range(ht_len):
            distances_matrix[i, j] = 1 - iou3d(gts_corner[i], hts_corner[j])
    distances_matrix[distances_matrix > max_iou] = np.nan
    return distances_matrix


class COLORS:
    """Color scheme for logging to console"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_dataloader(opt, dataset, shuffle):
    return DataLoader(
        dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=shuffle)


def cropPC(PC, box, offset=0, scale=1.0):
    box_tmp = copy.deepcopy(box)
    box_tmp.wlh = box_tmp.wlh * scale
    maxi = np.max(box_tmp.corners(), 1) + offset
    mini = np.min(box_tmp.corners(), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC


def cropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = cropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.rotation_matrix)
    trans = -new_box.center

    # align data
    new_PC.translate(trans)
    new_box.translate(trans)
    new_PC.rotate((rot_mat))
    new_box.rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = cropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC
