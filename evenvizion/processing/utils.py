"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

To describe utils functions.
"""

import json

import cv2
import numpy as np
import numpy.linalg as linalg

from evenvizion.processing.constants import INFINITY_COORDINATE, \
    THRESHOLD_FOR_FIND_HOMOGRAPHY, LENGTH_ACCOUNTED_POINTS


class HomographyException(Exception):
    """
        A class used to handle the exception.

        Attributes
        ----------
        message: : str
            Exception description.

        Methods
        ----------
    """

    def __init__(self, message="can't calculate homography matrix"):
        self.message = message
        super().__init__(message)


def remove_double_matching(pts_a, pts_b):
    """ Return only unique key points coordinates.

        Parameters
        ----------
        pts_a: np.array
            An array of x and y coordinates of each key point which matches with points from pts_b.

        pts_b: np.array
            An array of x and y coordinates of each key point which matches with points from pts_a.

        Returns
        ----------
        new_pts_a: np.array
            An array of unique matching points from pts_a.

        new_pts_b: np.array
            An array of unique matching points from pts_b.
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
    """ To apply homography transformation with matrix H to vector.

        Parameters
        ----------
         vector: np.array
            An array of x and y vector coordinates, vector[2] should be equal to 1.
                example vector = [21, 457, 1].

        matrix_H: np.array
            Homography matrix.

        Returns
        ----------
        np.array
            An array of x and y vector coordinates after homography transformation.

    """
    while len(vector) < 3:
        vector = np.append(vector, [1])
    new_vector = np.dot(matrix_H, vector)
    return new_vector[:-1] / new_vector[-1]


def inverse_homography_transformation(vector, matrix_H):
    """ To apply inverse homography transformation with matrix H to vector.

        Parameters
        ----------
         vector: np.array
            An array of x and y vector coordinates, vector[2] should be equal to 1.
                example vector = [21, 457, 1].

        matrix_H: np.array
            Homography matrix.

        Returns
        ----------
        np.array
            An array of x and y vector coordinates after inverse homography transformation.
    """
    while len(vector) < 3:
        vector = np.append(vector, [1])
    new_vector = np.dot(linalg.inv(matrix_H), vector)
    return new_vector[:-1] / new_vector[-1]


def matrix_superposition(H, matrix_H_superposition, matrix_H_first=False):
    """ To apply homography transformation with matrix H to vector.

        Parameters
        ----------
        H: np.array
            The first matrix for which we want to get superposition.

        matrix_H_superposition: np.array
            The second matrix for which we want to get superposition.


        matrix_H_first: bool, optional
            if you want to get H as a result then set True,
            if you want to multiply H and matrix_H_superposition then set False.

        Returns
        ----------
        np.array
            Matrix superposition.
    """
    if H is not None:
        if matrix_H_first:
            matrix_H_superposition = H
        else:
            matrix_H_superposition = np.dot(H, matrix_H_superposition)
            matrix_H_superposition = np.divide(matrix_H_superposition, matrix_H_superposition[2][2])
    return matrix_H_superposition


def read_homography_dict(path_to_homography_dict):
    """ To apply homography transformation with matrix H to vector.

        Parameters
        ----------
        path_to_homography_dict: str
            The path to the json with a homography dict.
                json format: {"frame_no": {"H":[]},
                              ...
                              "frame_no": {"H":[]},
                              "resize_info": {"h": (int), "w": (int))}}.
        Returns
        ----------
        homography_dict: dict
            A dict of homography matrix.
                format: {frame_no: {"H":[]},
                              ...
                          frame_no: {"H":[]}}.

        resize_info: dict
            An image shape which is used to get homography dictionary.
                format:{"h": (int), "w": (int))}.
    """
    with open(path_to_homography_dict, "r") as curr_json:
        homography_dict = json.load(curr_json)
        if "resize_info" in homography_dict:
            resize_info = homography_dict["resize_info"]
        else:
            raise ValueError("Specify the height and width of the frame "
                             "for which the homography matrix was obtained")
        resize_info = resize_info
        del homography_dict["resize_info"]
        homography_dict = {int(k): v for k, v in homography_dict.items()}
    return homography_dict, resize_info


def superposition_dict(homography_dict):
    """ To get dict of matrix superposition for each frame.

        Parameters
        ----------
        homography_dict: dict
            A dict of homography matrix.
                format: {frame_no: {"H":[]},
                              ...
                          frame_no: {"H":[]}}.
        Returns
        ----------
        superposition_homography_dict: dict
            A dictionary of matrix superposition for each frame.
                format: {frame_no: [],
                              ...
                          frame_no: []}.
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
    """ To check whether the coordinates are infinite.

        Parameters
        ----------
        coordinates_value: np.array, or list
            x and y coordinates you want to check.

        Returns
        ----------
        bool
            Are coordinates infinite or not.
    """
    try:
        return any(i >= INFINITY_COORDINATE for i in coordinates_value)
    except TypeError:
        return coordinates_value >= INFINITY_COORDINATE


def read_json_with_coordinates(path_to_coordinate):
    """ To read the json file with coordinate.

    Parameters
    ----------
    path_to_coordinate: str
        Path to json with homography dict.
            json format: {"frame_no": [{"x1":x, "y1": y}, ... {"x1":x, "y1": y}],
                          ...
                          "frame_no": [{"x1":x, "y1": y}, ... {"x1":x, "y1": y}]}.

    Returns
    ----------
    coordinates: dict
        A dictionary of coordinates.
            format: {frame_no: [{"x1":x, "y1": y}, ... {"x1":x, "y1": y}],
                     ...
                     frame_no: [{"x1":x, "y1": y}, ... {"x1":x, "y1": y}]}.
    """
    with open(path_to_coordinate, 'r') as path_to_detection:
        coordinates = json.load(path_to_detection)
        coordinates = {int(k): v for k, v in coordinates.items()}
    return coordinates


def get_largest_group_points(r_moving_dict, pts_a, pts_b):
    """ To read the json with coordinate.

    Parameters
    ----------
    r_moving_dict: dict
        A dictionary: key - displacement, value - point number which has this displacement.
            example: {0: [0, 2, 3, ...],
                     ...,
                     10: [1, 5, 7, ...].

    Returns
    ----------
    new_pts_a: np.array
        The coordinates of points which are located in key with the most number of value.
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


def find_point_displacement(matrix_H, pts_a, pts_b):
    """ To group points according to their displacement.

    Parameters
    ----------
    matrix_H: np.array
        Matrix describing the change between pts_a and pts_b coordinates.

    pts_a: np.array
        An array of x and y coordinates of each key point which matches with points from pts_b.

    pts_b: np.array
        An array of x and y coordinates of each key point which matches with points from pts_a.


    Returns
    ----------
    r_moving_dict: dict
        A dictionary: key - displacement, value - points number which has this displacement.
            example: {0: [0, 2, 3, ...],
                     ...,
                     10: [1, 5, 7, ...].
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
    return r_moving_dict


def compute_homography(pts_a, pts_b, matrix_H_prev=None):
    """ To compute matrix described the change between static_pts_a and static_pts_b coordinates.

    Parameters
    ----------

    pts_a: np.array
        An array of x and y coordinates of each key point which matches with points from pts_b.

    pts_b: np.array
        An array of x and y coordinates of each key point which matches with points from pts_a.

    matrix_H_prev: np.array, optimal
        If you want to get a matrix describing the change between coordinates in the origin plane,
        you should specify homography matrix superposition
        between origin and frame_a (pts_a located on frame_a).


    Returns
    ----------
    matrix_H_new:  np.array
        Matrix describing the change between static_pts_a and static_pts_b coordinates.
    """
    if matrix_H_prev is not None:
        pts_a = np.apply_along_axis(
            homography_transformation, 1, pts_a, matrix_H_prev)
        pts_b = np.apply_along_axis(
            homography_transformation, 1, pts_b, matrix_H_prev)
    (matrix_H_new, status_new) = cv2.findHomography(
        np.array(pts_a), np.array(pts_b),
        cv2.RANSAC, THRESHOLD_FOR_FIND_HOMOGRAPHY)
    if np.sum(status_new) < LENGTH_ACCOUNTED_POINTS * len(status_new):
        raise HomographyException("not enough points in homography calculation")
    if matrix_H_new is None:
        raise HomographyException()
    return matrix_H_new
