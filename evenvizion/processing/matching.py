"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

To describe the key points from a frame.
"""

import cv2
import numpy as np

from evenvizion.processing.constants import THRESHOLD_FOR_FIND_HOMOGRAPHY, \
    LOWES_RATIO, MINIMUM_MATCHING_POINTS
from evenvizion.processing.utils import remove_double_matching, \
    find_point_displacement, get_largest_group_points


class NoMatchesException(Exception):
    """
        The class used to handle the exception.

        Attributes
        ----------
        reason: : str
            Exception reason.

        description: str
            Exception description.

        Methods
        ----------
    """

    def __init__(self, reason, description="no matches found"):
        self.reason = reason
        self.description = description
        super().__init__(description)

    def __str__(self):
        return f'{self.reason} -> {self.description}'


class KeyPoints:
    """
        A class used to describe key points.

        Attributes
        ----------
        coordinates: : np.array
            An array of key points coordinates.

        descriptors: : np.array
            An array of key points descriptors.

        Methods
        -------
        match_kps(acceding_kps: KeyPoints)
            To match key points using their descriptors.
            Find 2 nearest neighbours for every key point,
            and using Lowe's ratio test to delete false matchings.

        match_static_kps(acceding_kps: KeyPoints)
            To match key points which are on a static object using their descriptors.

    """

    def __init__(self, coordinates, descriptors):
        self.coordinates = coordinates
        self.descriptors = descriptors

    def match_kps(self, acceding_kps, ratio=LOWES_RATIO, min_matching_pts=MINIMUM_MATCHING_POINTS):
        """ To match key points using their descriptors.
         Find 2 nearest neighbours for every key point,
         and using Lowe's ratio test to delete false matchings.

        Parameters
        ----------
        acceding_kps: KeyPoints
            Key points should be matched with self key points.

        ratio: float, optional
            The ratio for Lowe's test.

        min_matching_pts: int, optional
            The minimum number of matching points you want to find.

        Returns
        ----------
        pts_a: np.array
            An array of x and y coordinates of each self key point
            which matches with acceding key points.

        pts_b: np.array
            An array of x and y coordinates of each acceding key point
            which matches with self key points.
        """

        matcher = cv2.DescriptorMatcher_create(
            "BruteForce")
        if self.descriptors is None:
            raise NoMatchesException("self.descriptors is None", "couldn't process")
        if acceding_kps.descriptors is None:
            raise NoMatchesException("kps.descriptors is None", "couldn't process")
        raw_matches = matcher.knnMatch(self.descriptors, acceding_kps.descriptors, 2)
        if raw_matches is None:
            raise NoMatchesException("raw_matches is None",
                                     "can't find matches points between this frames")
        matches = lowes_ratio_test(raw_matches, ratio)
        if len(matches) < min_matching_pts:
            raise NoMatchesException("len(matches) {} < "
                                     "min_matching_pts {}".format(len(matches), min_matching_pts),
                                     "couldn't process")
        pts_a = np.float32([self.coordinates[i] for (_, i) in matches])
        pts_b = np.float32([acceding_kps.coordinates[i] for (i, _) in matches])
        pts_a, pts_b = remove_double_matching(pts_a, pts_b)
        if pts_a is None:
            raise NoMatchesException("problem with finding points at the first frame",
                                     "couldn't process")
        if pts_b is None:
            raise NoMatchesException("problem with finding points at the second frame",
                                     "couldn't process")
        if len(pts_a) != len(pts_b):
            raise NoMatchesException("lengths between matching points are different",
                                     "couldn't process")
        return pts_a, pts_b

    def match_static_kps(self, acceding_kps, reproj_thresh=THRESHOLD_FOR_FIND_HOMOGRAPHY):
        """ To match key points which are on static object using their descriptors.

        Parameters
        ----------
        acceding_kps: KeyPoints
            Key points should be matched with self key points.

        reproj_thresh: float, or int
            The threshold for OpenCV findHomography() function.

        Returns
        ----------
        static_pts_a: np.array
            An array of x and y coordinates of each self key point on a static object
            which can be matched with acceding key points.

        static_pts_b: np.array
            An array of x and y coordinates of each acceding key point on a static object
            which can be matched with self key points.
        """
        matching_keypoints = self.match_kps(acceding_kps)
        if matching_keypoints is None:
            raise NoMatchesException("can't find matching points", "couldn't process")
        pts_a, pts_b = matching_keypoints
        (matrix_H, _) = cv2.findHomography(np.array(pts_a), np.array(pts_b),
                                           cv2.RANSAC, reproj_thresh)
        if matrix_H is None:
            raise NoMatchesException("can't find homography matrix", "couldn't process")

        r_moving_dict = find_point_displacement(matrix_H, pts_a, pts_b)
        static_pts_a, static_pts_b = get_largest_group_points(r_moving_dict, pts_a, pts_b)
        return static_pts_a, static_pts_b


def lowes_ratio_test(raw_matches, ratio=LOWES_RATIO):
    """ To delete false matchings using Lowe's ratio test.

    Parameters
    ----------
    raw_matches: list
        A list of key points with 2 nearest (similar descriptors) neighbours.

    ratio: float, or int
        The ratio for Lowe's test.

    Returns
    ----------
    static_pts_a: np.array
        An array of x and y coordinates of each self key point on a static object
        which can be matched with acceding key points.

    static_pts_b: np.array
        An array of x and y coordinates of each acceding key point on a static object
        which can be matched with self key points.
   """
    train_idx_dict = {}
    query_idx_dict = {}
    for matches in raw_matches:
        if len(matches) == 2 and matches[0].distance < matches[1].distance * ratio:
            if train_idx_dict.get(matches[0].trainIdx) is None:
                train_idx_dict[matches[0].trainIdx] = []
            train_idx_dict[matches[0].trainIdx].append(matches[0].queryIdx)
            if query_idx_dict.get(matches[0].queryIdx) is None:
                query_idx_dict[matches[0].queryIdx] = []
            query_idx_dict[matches[0].queryIdx].append(matches[0].trainIdx)
    matches = filter_corresponding_points(train_idx_dict, query_idx_dict)
    return matches


def filter_corresponding_points(train_idx_dict, query_idx_dict):
    """ If a point has two matchers then delete this point from consideration.

    Parameters
    ----------
    train_idx_dict: dict
        key - matching points number from the self frame,
        value - matching points number from the acceding frame
            example: {1: [10], 5: [13, 11], ...}
            It means that the self key point №1 was matched with the acceding key point №10.
            The №5 was matched with 2 points, it means that it will be removed.

    query_idx_dict: float, or int
        key - matching points number from the acceding frame,
        value - matching points number from the self frame
            example: {10: [1], 13: [5, 7], ...}
            It means that the acceding key point №10 was matched with the self key point №1.
            The №13 was matched with 2 points, it means that it will be removed.

    Returns
    ----------
    matches: list
        A list of matching points without duplication.

    """
    k_for_del = []
    matches = []
    for key, value in train_idx_dict.items():
        if len(value) > 1:
            k_for_del.append(key)
    for key, value in query_idx_dict.items():
        if len(value) > 1:
            for i in value:
                k_for_del.append(i)
    for i in set(k_for_del):
        del train_idx_dict[i]
    for train_idx, query_idx in train_idx_dict.items():
        matches.append((train_idx, query_idx[0]))
    return matches
