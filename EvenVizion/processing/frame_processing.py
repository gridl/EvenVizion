from EvenVizion.processing.matching import *
from EvenVizion.processing.matching import KeyPoints
import imutils
import logging
INFINITY_COORDINATE_FLAG = 10000
LENGTH_ACCOUNTED_POINTS = 0.7

logging.basicConfig(level=logging.DEBUG)
THRESHOLD_FOR_FIND_HOMOGRAPHY = 3.0





class FrameProcessing:
    """
    class for find homography transformation for all frames
    """

    def __init__(self, frame, features_type_list=None):
        self.isv3 = imutils.is_cv3(or_better=True)
        self.frame = frame
        self.features_types = features_type_list
        if self.features_types is None:
            self.features_types = ["ORB", "SIFT", "SURF"]

    def detect_and_describe_features(self, features_name):
        """
        Find singular point and their descriptions
        """
        if features_name == "ORB":
            orb = cv2.ORB_create()
            (coords, descriptors) = orb.detectAndCompute(self.frame, None)
        elif features_name == "SIFT":
            sift = cv2.xfeatures2d.SIFT_create()
            (coords, descriptors) = sift.detectAndCompute(self.frame, None)
        elif features_name == "SURF":
            surf = cv2.xfeatures2d.SURF_create(extended=1, hessianThreshold=400)
            (coords, descriptors) = surf.detectAndCompute(self.frame, None)
        else:
            raise ValueError("You need to choose descriptors type")
        coords = np.float32([kp.pt for kp in coords])
        return (coords, descriptors)

    def concatenate_all_features_types(self, acceding_image):
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
            return None, None
        all_static_pts_a, all_static_pts_b = remove_double_matching(all_static_pts_a_with_repetitions,
                                                              all_static_pts_b_with_repetitions)
        return all_static_pts_a, all_static_pts_b

    def stitch(self, acceding_image):
        """
        Get homography matrix from 2 frames
        """
        static_pts_a, static_pts_b = \
            self.concatenate_all_features_types(acceding_image)
        if static_pts_a is None or static_pts_b is None:
            raise NoMatchesException("can't find keypoints that lie on static objects ", "couldn't process")
        return static_pts_a, static_pts_b
