# здесь работаем только с точками
from EvenVizion.processing.frame_processing import *

from EvenVizion.processing.utils import *

LOWES_RATIO = 0.5
MINIMUM_MATCHING_POINTS = 4
THRESHOLD_FOR_FIND_HOMOGRAPHY = 3.0


class NoMatchesException(Exception):
    def __init__(self, reason, description="no matches found"):
        self.reason = reason
        self.description = description
        super().__init__(description)

    def __str__(self):
        return f'{self.reason} -> {self.description}'


class KeyPoints:

    def __init__(self, coords, descriptors):
        self.coords = coords
        self.descriptors = descriptors

    def match_kps(self, acceding_kps, ratio=LOWES_RATIO, min_matching_pts=MINIMUM_MATCHING_POINTS):
        matcher = cv2.DescriptorMatcher_create(
            "BruteForce")
        if self.descriptors is None:
            raise NoMatchesException("self.descriptors is None", "couldn't process")
        if acceding_kps.descriptors is None:
            raise NoMatchesException("kps.descriptors is None", "couldn't process")
        raw_matches = matcher.knnMatch(self.descriptors, acceding_kps.descriptors, 2)
        if raw_matches is None:
            raise NoMatchesException("raw_matches is None", "can't find matches points between this frames")
        matches = lowes_ratio_test(raw_matches, ratio)
        if len(matches) < min_matching_pts:
            raise NoMatchesException("len(matches) {} < min_matching_pts {}".format(len(matches), min_matching_pts),
                                     "couldn't process")
        pts_a = np.float32([self.coords[i] for (_, i) in matches])
        pts_b = np.float32([acceding_kps.coords[i] for (i, _) in matches])
        pts_a, pts_b = remove_double_matching(pts_a, pts_b)
        if pts_a is None:
            raise NoMatchesException("problem with finding points at the first frame", "couldn't process")
        if pts_b is None:
            raise NoMatchesException("problem with finding points at the second frame", "couldn't process")
        if len(pts_a) != len(pts_b):
            raise NoMatchesException("lengths between matching points are different", "couldn't process")
        return pts_a, pts_b

    def match_static_kps(self, acceding_kps, reproj_thresh=THRESHOLD_FOR_FIND_HOMOGRAPHY):
        matching_keypoints = self.match_kps(acceding_kps)
        if matching_keypoints is None:
            raise NoMatchesException("can't find matching points", "couldn't process")
        pts_a, pts_b = matching_keypoints
        (matrix_H, status) = cv2.findHomography(np.array(pts_a), np.array(pts_b),
                                                cv2.RANSAC, reproj_thresh)
        if matrix_H is None:
            raise NoMatchesException("can't find homography matrix", "couldn't process")
        static_pts_a, static_pts_b = find_static_groups(matrix_H, pts_a, pts_b)
        return static_pts_a, static_pts_b


def lowes_ratio_test(raw_matches, ratio=LOWES_RATIO):
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
    """
    If we have points on image a which have two corresponding point on image b, we delete it from processing
    The same works vice versa
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
