import json

import cv2
import numpy as np
import numpy.linalg as linalg

INFINITY_COORDINATE = 10000
THRESHOLD_FOR_FIND_HOMOGRAPHY = 3.0
LENGTH_ACCOUNTED_POINTS = 0.7


class HomographyException(Exception):
    def __init__(self, message="can't calculate homography matrix"):
        self.message = message
        super().__init__(message)


def remove_double_matching(pts_a, pts_b):
    """
    Sort matching points for duplicate removing
    """
    matching_dict = {}
    new_pts_a = []
    new_pts_b = []
    for i, _ in enumerate(pts_a):
        matching_dict[(pts_a[i][0], pts_a[i][1])] = pts_b[i]
    for key, value in matching_dict.items():
        new_pts_a.append(np.array(key))
        new_pts_b.append(value)
    return new_pts_a, new_pts_b


def homography_transformation(vector, matrix_H):
    """
    applying homography transformation with matrix H
    """
    if len(vector) < 3:
        vector = np.append(vector, [1])
    new_vector = np.dot(matrix_H, vector)
    return new_vector[:-1] / new_vector[-1]


def inverse_homography_transformation(a, H):
    if len(a) < 3:
        a = np.append(a, [1])
    new_vector = np.dot(linalg.inv(H), a)
    return (new_vector[:-1] / new_vector[-1])


def matrix_superposition(H, matrix_H_superposition, matrix_H_first=False):
    """
    get matrixes superposition
    """
    if H is not None:
        if matrix_H_first:
            matrix_H_superposition = H
        else:
            matrix_H_superposition = np.dot(H, matrix_H_superposition)
            matrix_H_superposition = np.divide(matrix_H_superposition, matrix_H_superposition[2][2])
    return matrix_H_superposition


def read_homography_dict(path_to_homography_dict):
    """
    read homography dict from the json file.
    json file format:
    {"frame_no": {"H":[]},
     ...
     "frame_no": {"H":[]},
     "resize_info": {"h": (int), "w": (int))}
    }
    """
    with open(path_to_homography_dict, "r") as curr_json:
        homography_dict = json.load(curr_json)
        if "resize_info" in homography_dict:
            resize_info = homography_dict["resize_info"]
        else:
            raise ValueError("Specify the height and width of the frame for which the homography matrix was obtained")
        resize_info = resize_info
        del homography_dict["resize_info"]
        homography_dict = {int(k): v for k, v in homography_dict.items()}
    return homography_dict, resize_info


def superposition_dict(homography_dict):
    """
    Get matrix superposition for each frame
    """
    superposition_homography_dict = {}
    matrix_H_next = None
    H_first = True
    superposition_homography_dict[1] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for frame_no, frame_H in homography_dict.items():
        matrix_H_next = matrix_superposition(frame_H["H"], matrix_H_next, H_first)
        H_first = False
        superposition_homography_dict[frame_no] = matrix_H_next
    return superposition_homography_dict


def are_infinity_coordinates(coordinates_value):
    return (coordinates_value[0] > INFINITY_COORDINATE or coordinates_value[1] > INFINITY_COORDINATE)


def get_largest_group_points(r_moving_dict, pts_a, pts_b):
    """
    """
    new_pts_a = []
    new_pts_b = []
    k_max_len = None
    max_len = 0
    for key, value in r_moving_dict.items():
        if len(value) > max_len:
            max_len = len(value)
            k_max_len = key
    for i in r_moving_dict[k_max_len]:
        new_pts_a.append(pts_a[i])
        new_pts_b.append(pts_b[i])
    return np.array(new_pts_a), np.array(new_pts_b)


def find_static_groups(matrix_H, pts_a, pts_b):
    """
    For more accurate assessment of the camera homography the
    points shouldn't be placed on the moving objects, such as persons, cars, etc.
    """
    r_moving = []
    r_moving_dict = {}
    if len(pts_a) != len(pts_b):
        raise ValueError("in find_static_part, len(pts_a) != len(pts_b)")
    for i, _ in enumerate(pts_a):
        new_vector = (pts_a[i][0], pts_a[i][1], 1)
        transform_vector = (np.dot(matrix_H, new_vector))
        r_distance = round(np.sum(np.subtract(transform_vector[:2]
                                              / transform_vector[2], pts_b[i]) ** 2) ** 0.5)
        r_moving.append(r_distance)
        if r_moving_dict.get(r_distance) is None:
            r_moving_dict[r_distance] = []
        r_moving_dict[r_distance].append(i)
    static_pts_a, static_pts_b = get_largest_group_points(r_moving_dict, pts_a, pts_b)
    return np.array(static_pts_a), np.array(static_pts_b)


def compute_homography(static_pts_a, static_pts_b, matrix_H_prev=None):
    if matrix_H_prev is not None:
        static_pts_a = np.apply_along_axis(
            homography_transformation, 1, static_pts_a, matrix_H_prev)
        static_pts_b = np.apply_along_axis(
            homography_transformation, 1, static_pts_b, matrix_H_prev)
    (matrix_H_new, status_new) = cv2.findHomography(
        np.array(static_pts_a), np.array(static_pts_b),
        cv2.RANSAC, THRESHOLD_FOR_FIND_HOMOGRAPHY)
    if np.sum(status_new) < LENGTH_ACCOUNTED_POINTS * len(status_new):
        raise HomographyException("not enough points in homography calculation")
    if matrix_H_new is None:
        raise HomographyException()
    return matrix_H_new
