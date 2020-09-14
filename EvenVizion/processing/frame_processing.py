"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion%20-%20video%20based%20camera%20localization%20component.pdf

This is licensed under an MIT license. See the LICENSE.md file
for more information.

To describe a frame with the help of key points of different types.
"""
import imutils
import cv2
import numpy as np

from EvenVizion.processing.matching import KeyPoints, NoMatchesException
from EvenVizion.processing.utils import remove_double_matching


class FrameProcessing:
    """
    A class used to describe a frame with the help of key points of different types.

    Attributes
    ----------
    frame: np.array
        An array of n special points from the first frame.

    Methods
    -------
    detect_and_describe_features(frame)
        To find key points on a frame and return their coordinates and descriptors.
    concatenate_all_features_types(frame)
        To concatenate all key points types in one array.
    """

    def __init__(self, frame, features_type_list=None):
        self.isv3 = imutils.is_cv3(or_better=True)
        self.frame = frame
        self.features_types = features_type_list or ["SURF", "SIFT", "ORB"]

    def detect_and_describe_features(self, features_name):
        """ To find key points on a frame and return their coordinates and descriptors.

        Parameters
        ----------
        features_name: str
            The name of features type.
                example: "ORB"

        Returns
        ----------
        coordinates: np.array
            An array of x and y coordinates of each key point.

        descriptors: np.array
            An array of descriptors of each key point.
        """
        if features_name == "ORB":
            orb = cv2.ORB_create()
            (coordinates, descriptors) = orb.detectAndCompute(self.frame, None)
        elif features_name == "SIFT":
            sift = cv2.xfeatures2d.SIFT_create()
            (coordinates, descriptors) = sift.detectAndCompute(self.frame, None)
        elif features_name == "SURF":
            surf = cv2.xfeatures2d.SURF_create(extended=1, hessianThreshold=400)
            (coordinates, descriptors) = surf.detectAndCompute(self.frame, None)
        else:
            raise ValueError("You need to choose descriptors type")
        coordinates = np.float32([kp.pt for kp in coordinates])
        return coordinates, descriptors

    def concatenate_all_features_types(self, acceding_image):
        """ To concatenate all key points types in one array.

        Parameters
        ----------
        acceding_image: np.array
            The frame to concatenate.

        Returns
        ----------
        all_matching_pts_a: np.array
            An array of x and y coordinates of key points on static objects from self.frame.

        all_matching_pts_b: np.array
            An array of x and y coordinates of key points on static objects from an acceding image.
        """
        all_static_pts_a_with_repetitions = []
        all_static_pts_b_with_repetitions = []
        for feature_type in self.features_types:
            (coords_a, descriptors_a) = self.detect_and_describe_features(feature_type)
            (coords_b, descriptors_b) = acceding_image.detect_and_describe_features(feature_type)
            kps_a = KeyPoints(coords_a, descriptors_a)
            kps_b = KeyPoints(coords_b, descriptors_b)
            static_pts_a, static_pts_b = kps_a.match_static_kps(kps_b)
            all_static_pts_a_with_repetitions.extend(static_pts_a)
            all_static_pts_b_with_repetitions.extend(static_pts_b)
        if all_static_pts_a_with_repetitions is None or all_static_pts_b_with_repetitions is None:
            raise NoMatchesException("can't find keypoints that lie on static objects ",
                                     "couldn't process")
        all_matching_pts_a, all_matching_pts_b = \
            remove_double_matching(all_static_pts_a_with_repetitions,
                                   all_static_pts_b_with_repetitions)
        if all_static_pts_a_with_repetitions is None or all_static_pts_b_with_repetitions is None:
            raise NoMatchesException("can't find keypoints that lie on static objects ",
                                     "couldn't process")
        return all_matching_pts_a, all_matching_pts_b
