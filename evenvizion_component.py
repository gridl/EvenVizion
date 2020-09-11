"""
main script for find homography matrix and visualize fixed coordinate system
"""
import argparse
import json
import logging
import os
from copy import deepcopy
import cv2
import imutils
import numpy as np

logging.basicConfig(level=logging.INFO)

INFINITY_COORDINATE_FLAG = 10000
LOWES_RATIO = 0.5
THRESHOLD_FOR_FIND_HOMOGRAPHY = 3.0
LENGTH_ACCOUNTED_POINTS = 0.7


def homography_transformation(vector_a, matrix_H):
    """
    applying homography transformation with matrix H
    """
    if len(vector_a) < 3:
        vector_a = np.append(vector_a, [1])
    new_vector = np.dot(matrix_H, vector_a)
    return new_vector[:-1] / new_vector[-1]


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


class Stitcher:
    """
    class for find homography transformation for all frames
    """

    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
        self.characteristic_dict = {}
        self.counter = 0
        self.ratio = LOWES_RATIO
        self.reproj_thresh = THRESHOLD_FOR_FIND_HOMOGRAPHY
        self.min_length_accounted_points = LENGTH_ACCOUNTED_POINTS

    def get_one_type_features_for_analyze(self, image_a, image_b, features_name):
        """
        Get points and there descriptors to analyze camera moving
        """
        (kps_a, features_a) = self.detect_and_describe_features(image_a, features_name)
        (kps_b, features_b) = self.detect_and_describe_features(image_b, features_name)
        matching_keypoints_result = self.match_keypoints(kps_a, kps_b, features_a, features_b)
        if matching_keypoints_result is None:
            logging.info("Problem with points matching")
            return None, None, None, None
        _, matrix_H, _, pts_a, pts_b = matching_keypoints_result
        if matrix_H is None:
            logging.info("len(matches)G<4, can't find Homography")
            return None, None, None, None
        new_pts_a, new_pts_b = find_static_part(matrix_H, pts_a, pts_b)
        return new_pts_a, new_pts_b, pts_a, pts_b

    def get_features_for_analyze(self, image_a,
                                 image_b):
        """
        Get different type os pictures features
        """
        static_pts_a_orb, static_pts_b_orb, pts_a_orb, pts_b_orb = \
            self.get_one_type_features_for_analyze(image_a, image_b, features_name="ORB")
        static_pts_a_sift, static_pts_b_sift, pts_a_sift, pts_b_sift = \
            self.get_one_type_features_for_analyze(image_a, image_b, features_name="SIFT")
        static_pts_a_surf, static_pts_b_surf, pts_a_surf, pts_b_surf = \
            self.get_one_type_features_for_analyze(image_a, image_b, features_name="SURF")
        if static_pts_a_orb is None or static_pts_a_sift is None or static_pts_a_surf is None:
            return None, None, None
        # stack all features
        matching_dict = {}
        pts_a = np.vstack((pts_a_surf, pts_a_sift, pts_a_orb))
        pts_b = np.vstack((pts_b_surf, pts_b_sift, pts_b_orb))
        static_pts_a_with_repetition = \
            np.vstack((static_pts_a_surf, static_pts_a_sift, static_pts_a_orb))
        static_pts_b_with_repetition = \
            np.vstack((static_pts_b_surf, static_pts_b_sift, static_pts_b_orb))
        static_pts_a = []
        static_pts_b = []
        ###

        # finaly matching
        for i, _ in enumerate(static_pts_a_with_repetition):
            matching_dict[(static_pts_a_with_repetition[i][0],
                           static_pts_a_with_repetition[i][1])] = static_pts_b_with_repetition[i]
        for key, value in matching_dict.items():
            static_pts_a.append(np.array(key))
            static_pts_b.append(value)
        pts_visualization = self.draw_static_matches(image_a, image_b, pts_a, pts_b)
        return static_pts_a, static_pts_b, pts_visualization

    def stitch(self, images, matrix_H_prev,
               show_matches=False):
        """
        Get homography matrix from 2 frames
        """
        (image_b, image_a) = images
        static_pts_a, static_pts_b, pts_visualization = \
            self.get_features_for_analyze(image_a, image_b)
        static_pts_a_draw, static_pts_b_draw = static_pts_a, static_pts_b
        if static_pts_a is None or len(static_pts_a_draw) < args.number_of_matching_points:
            logging.info("Homography matrix can't be calculated, "
                         "because there are no matching points between this frames")
            return None, None, None
        if matrix_H_prev is not None:
            static_pts_a = np.apply_along_axis(
                homography_transformation, 1, (static_pts_a), matrix_H_prev)
            static_pts_b = np.apply_along_axis(
                homography_transformation, 1, (static_pts_b), matrix_H_prev)

        # compute the homography between the two sorted sets of points
        (matrix_H_new, status_new) = cv2.findHomography(
            np.array(static_pts_a), np.array(static_pts_b),
            cv2.RANSAC, self.reproj_thresh)  # H-transformation matrix
        if np.sum(status_new) < self.min_length_accounted_points * len(status_new):
            logging.debug("in static part only %s ", np.sum(status_new))
            logging.debug("all pts %s", len(status_new))
            return None, pts_visualization, None
        if matrix_H_new is None:
            logging.debug("H is None with static pts")
            return None, pts_visualization, None
        if show_matches:
            vis_2 = self.draw_static_matches(image_a, image_b, static_pts_a_draw, static_pts_b_draw)
            return (matrix_H_new, pts_visualization, vis_2)
        return matrix_H_new, None, None

    def detect_and_describe_features(self, image, features_name):
        """
        Find singular point and their descriptions
        """

        if features_name == "ORB":
            orb = cv2.ORB_create()
            (kps, features) = orb.detectAndCompute(image, None)
        elif features_name == "SIFT":
            sift = cv2.xfeatures2d.SIFT_create()
            (kps, features) = sift.detectAndCompute(image, None)
        elif features_name == "SURF":
            surf = cv2.xfeatures2d.SURF_create(extended=1, hessianThreshold=400)
            (kps, features) = surf.detectAndCompute(image, None)
        else:
            raise "You need to choose features type"
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def delete_duplicate_in_matching(self, train_idx_dict, query_idx_dict):
        """
        Delete duplicate in matching
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

    def get_matching_points_without_duplicate(self, matches, kps_a, kps_b):
        """
        Sort matching points for duplicate removing
        """
        matching_dict = {}
        new_pts_a = []
        new_pts_b = []
        if len(matches) > args.number_of_matching_points:
            # for compute homography matrix should at least 4 points
            pts_a = np.float32([kps_a[i] for (_, i) in matches])
            pts_b = np.float32([kps_b[i] for (i, _) in matches])
            for i, _ in enumerate(pts_a):
                matching_dict[(pts_a[i][0], pts_a[i][1])] = pts_b[i]
            for key, value in matching_dict.items():
                new_pts_a.append(np.array(key))
                new_pts_b.append(value)
            return new_pts_a, new_pts_b
        return None, None

    def delete_matching_falses(self):
        """
        delete matching falses
        """
        train_idx_dict = {}
        query_idx_dict = {}
        for matches in self.raw_matches:
            if len(matches) == 2 and matches[0].distance < matches[1].distance * self.ratio:
                if train_idx_dict.get(matches[0].trainIdx) is None:
                    train_idx_dict[matches[0].trainIdx] = []
                train_idx_dict[matches[0].trainIdx].append(matches[0].queryIdx)
                if query_idx_dict.get(matches[0].queryIdx) is None:
                    query_idx_dict[matches[0].queryIdx] = []
                query_idx_dict[matches[0].queryIdx].append(matches[0].trainIdx)
        matches = self.delete_duplicate_in_matching(train_idx_dict, query_idx_dict)
        return matches

    def match_keypoints(self, kps_a, kps_b, features_a, features_b):
        """
        Brute Force keypoints matching
        """
        matcher = cv2.DescriptorMatcher_create(
            "BruteForce")
        if features_a is None or features_b is None:
            logging.debug("features_a is None or features_b is None")
            return None
        self.raw_matches = matcher.knnMatch(features_a, features_b, 2)
        if self.raw_matches is None:
            logging.debug("raw_matches is None")
            return None
        matches = self.delete_matching_falses()
        new_pts_a, new_pts_b = self.get_matching_points_without_duplicate(matches, kps_a, kps_b)
        if new_pts_a is not None and len(new_pts_a) == len(new_pts_b):
            (matrix_H, status) = cv2.findHomography(np.array(new_pts_a), np.array(new_pts_b),
                                                    cv2.RANSAC, self.reproj_thresh)
            return (matches, matrix_H, status, new_pts_a, new_pts_b)
        if new_pts_a is None:
            logging.debug("new_pts_a is None")
            return None
        if len(new_pts_a) != len(new_pts_b):
            logging.debug("len(new_pts_a) != len(new_pts_b)")
            return None
        return None

    def draw_static_matches(self, image_a, image_b, pts_a, pts_b):
        """
        Visualize only static keypoints
        """
        (h_a, w_a) = image_a.shape[:2]
        (h_b, w_b) = image_b.shape[:2]
        visualization = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype="uint8")
        visualization[0:h_a, 0:w_a] = image_a
        visualization[0:h_b, w_a:] = image_b
        counter = 0
        for (a_point, b_point) in zip(pts_a, pts_b):
            counter += 1
            pt_a = (int(a_point[0]), int(a_point[1]))
            pt_b = (int(b_point[0]) + w_a, int(b_point[1]))
            cv2.line(visualization, pt_a, pt_b, (0, 255, 0), 1)
        return visualization


###############filter functions
def get_static_pts(r_moving_dict, pts_a, pts_b):
    """
     Filter pts for getting only static ones
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


def find_static_part(matrix_H, pts_a, pts_b):
    """
    For more adequate assessment of the camera homography the
    poinsts schouldn't be placed on the moving objects, such as persons, cars, etc.
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
    new_pts_a, new_pts_b = get_static_pts(r_moving_dict, pts_a, pts_b)
    return np.array(new_pts_a), np.array(new_pts_b)


###################


def heatmap_visualization_with_coordinates(
        path_to_dict_with_homography_matrix, picture_template, capture):
    """
    Visualize heatmap with fixed coordinate system using all points in frame
    """
    with open(path_to_dict_with_homography_matrix, 'r') as homography__:
        dict_with_homography_matrix = json.load(homography__)
        del dict_with_homography_matrix["resize_info"]
        dict_with_homography_matrix = {int(k): v for k, v in dict_with_homography_matrix.items()}
    new_coordinat_dict = {}
    matrix_H_next = None
    matrix_H_first = True
    max_r = []
    logging.info("Start heatmap visualization")
    for (frame_no, frame_H) in dict_with_homography_matrix.items():
        new_coordinat_dict[frame_no] = []
        matrix_H_next = matrix_superposition(frame_H["H"], matrix_H_next, matrix_H_first)
        matrix_H_first = False
        new_pictures = np.apply_along_axis(
            homography_transformation, 2, (picture_template), matrix_H_next)
        _, original_image = capture.read()
        heatmap_visualization(new_pictures.tolist(), original_image, args.save_folder, frame_no)
        max_r.append(np.max(new_pictures))
    del dict_with_homography_matrix
    return np.max(max_r)


def make_template(shape):
    """
    Make array with each pixels coordinate
    """
    pictures_template = np.array([[[0] * shape[2]] * shape[1]] * shape[0])
    for i in range(shape[0]):
        for j in range(shape[1]):
            pictures_template[i][j] = [j, i, 1]
    return pictures_template


def part_line_visualization(r_movemet, image):
    """
    Function for line and new coordinate drawing on frame
    """
    height, width = image.shape[:-1]
    max_i = args.number_of_horizontal_lines
    max_j = args.number_of_vertical_lines
    for i in range(0, max_i):
        for j in range(0, max_j):
            cv2.rectangle(image, (int(j * width / max_j), int(i * height / max_i)),
                          (int((j + 1) * width / max_j), int((i + 1) * height / max_i)),
                          (0, 0, 0), 1)
            center_r_1 = [
                np.array([int((j * (width) + width / 2) / max_j),
                          int((i * (height) + height / 2) / max_i)]),
                np.array([int((j * (width) + width / 2) / max_j),
                          (int((i * (height) + height / 2) / max_i))]) +
                np.sign(r_movemet[(int((i * (height) + height / 2) / max_i))]
                        [int((j * (width) + width / 2) / max_j)][:-1]) * [10, 0]]
            center_r_2 = [
                np.array([int((j * (width) + width / 2) / max_j),
                          int((i * (height) + height / 2) / max_i)]),
                np.array([int((j * (width) + width / 2) / max_j),
                          (int((i * (height) + height / 2) / max_i))]) +
                np.sign(r_movemet[(int((i * (height) + height / 2) / max_i))]
                        [int((j * (width) + width / 2) / max_j)][:-1]) * [0, 10]]

            cv2.arrowedLine(image, (int(center_r_1[0][0]), int(center_r_1[0][1]) - 5),
                            (int(center_r_1[1][0]), int(center_r_1[1][1] - 5)), (0, 0, 0), 1)
            cv2.arrowedLine(image, (int(center_r_2[0][0]), int(center_r_2[0][1]) - 5),
                            (int(center_r_2[1][0]), int(center_r_2[1][1] - 5)), (0, 0, 0), 1)
            if r_movemet[int((i * (height) + height / 2) /
                             max_i)][int((j * (width) + width / 2) / max_j)][0] > \
                    INFINITY_COORDINATE_FLAG or \
                    r_movemet[int((i * (height) + height / 2) /
                                  max_i)][int((j * (width) + width / 2) / max_j)][1] > \
                    INFINITY_COORDINATE_FLAG:
                x_coordinate = "n/a"
                y_coordinate = "n/a"
            else:
                x_coordinate = round(
                    r_movemet[int((i * (height) + height / 2) / max_i)]
                    [int((j * (width) + width / 2) / max_j)][0])
                y_coordinate = round(
                    r_movemet[int((i * (height) + height / 2) / max_i)]
                    [int((j * (width) + width / 2) / max_j)][1])
            cv2.putText(image,
                        'x {}'.format(x_coordinate),
                        (int((j * (width) + width / 2) / max_j - 20),
                         int((i * (height) + height / 2) / max_i + 10)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0),
                        fontScale=1, thickness=1)
            cv2.putText(image,
                        'y {}'.format(y_coordinate),
                        (int((j * (width) + width / 2) / max_j - 10),
                         int((i * (height) + height / 2) / max_i - 7)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0),
                        fontScale=1, thickness=1)
    return image


def heatmap_visualization(new_cordinats, original_image, save_folder,
                          frame_no):
    """
    heatmap visualization
    """
    heatmap_save_folder = "{}/heatmap_visualization".format(save_folder)
    if not os.path.exists(heatmap_save_folder):
        os.mkdir(heatmap_save_folder)
    path_to_save = "{}/img{}.png".format(heatmap_save_folder, "%06d" % frame_no)
    image = imutils.resize(original_image, width=args.resize_width)
    heatmap = np.power(np.sum(np.power(new_cordinats, 2), axis=-1), 0.5)
    heatmap /= args.heatmap_constant
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + image
    part_line_visualization(new_cordinats, superimposed_img)
    cv2.imwrite(path_to_save, superimposed_img)


def get_json_with_homography_matrix(save_folder, capture):
    """
    Main function for getting json with homography matrix for each frames
    """
    success, result_image = capture.read()
    if not success:
        raise ValueError("Problem with video! Can't read first frame")
    result_image = deepcopy(imutils.resize(result_image, width=args.resize_width))
    i = 1
    matrix_H_superposition = None
    matrix_H_prev = None
    matrix_H_first = True
    stitcher = Stitcher()
    logging.info("Start video analysis")
    while success:
        i += 1
        image_a = deepcopy(result_image)
        success, image_b = capture.read()
        if not success:
            continue
        image_b = imutils.resize(image_b, width=args.resize_width)
        (matrix_H, vis, vis_2) = stitcher.stitch([image_a, image_b], matrix_H_superposition,
                                                 show_matches=args.show_matches)
        result_image = deepcopy(image_b)

        # matrix_H is None case processing
        if matrix_H is None:
            if args.none_H_processing:
                matrix_H = matrix_H_prev
            else:
                stitcher.characteristic_dict[i] = {"H": None}
                if args.show_matches:
                    cv2.imwrite("{}/vis_{}.png".format(save_folder, i), vis)
                    cv2.imwrite("{}/vis_filter_points_{}.png".format(save_folder, i), vis_2)
                continue
        ###

        stitcher.characteristic_dict[i] = {"H": matrix_H.tolist()}
        matrix_H_superposition = \
            matrix_superposition(matrix_H, matrix_H_superposition, matrix_H_first)
        matrix_H_prev = matrix_H
        matrix_H_first = False
        if args.show_matches:
            cv2.imwrite("{}/vis_{}.png".format(save_folder, i), vis)
            cv2.imwrite("{}/vis_filter_points_{}.png".format(save_folder, i), vis_2)
    stitcher.characteristic_dict["resize_info"] = \
        {"h": result_image.shape[0], "w": result_image.shape[1]}
    with open("{}/dict_with_homography_matrix.json".format(save_folder), "w") as json_:
        json.dump(stitcher.characteristic_dict, json_)
    return result_image


def coordinates_recalculation(original_coordinates, homography_dict):
    """
    Recalculate original coordinates into fixed coordinate system
    """
    recalculated_detection_coordinates = {}
    args.h_resize_coefficient = args.fixed_cordinat_resize_h / args.img_h
    args.w_resize_coefficient = args.fixed_cordinat_resize_w / args.img_w
    for frame_no, frame_info in original_coordinates.items():
        recalculated_detection_coordinates[frame_no] = []
        for rect in frame_info:
            new_rect = deepcopy(rect)
            [new_rect["x1"], new_rect["y1"]] = np.around(homography_transformation(
                [args.w_resize_coefficient * rect["x1"], args.h_resize_coefficient * rect["y1"]],
                homography_dict[frame_no]), decimals=2)
            recalculated_detection_coordinates[frame_no].append(new_rect)
    return recalculated_detection_coordinates


def reformat_homography_dict(path_to_gomography_dict):
    """
    Get matrix superposition for each frame
    """
    with open(path_to_gomography_dict, "r") as curr_json:
        homography_dict = json.load(curr_json)
        resize_info = homography_dict["resize_info"]
        args.fixed_cordinat_resize_h = resize_info["h"]
        args.fixed_cordinat_resize_w = resize_info["w"]
        del homography_dict["resize_info"]
        homography_dict = {int(k): v for k, v in homography_dict.items()}
    superposition_homography_dict = {}
    matrix_H_next = None
    H_first = True
    superposition_homography_dict[1] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for frame_no, frame_H in homography_dict.items():
        matrix_H_next = matrix_superposition(frame_H["H"], matrix_H_next, H_first)
        H_first = False
        superposition_homography_dict[frame_no] = matrix_H_next
    return superposition_homography_dict


def visualize_comparison_original_with_fixed_coordinate(capture, original_coordinates,
                                                        recalculated_coordinates):
    """
    Visualize two coordinate system on one frame
    """
    logging.info("Start visualization comparison original with fixed coordinate")
    comparison_save_folder = "{}/fixed_coordinate_system_visualization".format(args.save_folder)
    success, _ = capture.read()
    frame_no = 1
    while success:
        success, image = capture.read()
        frame_no += 1
        if not success:
            continue
        for (rect_original, rect_fixed) in \
                zip(original_coordinates[frame_no], recalculated_coordinates[frame_no]):
            cv2.circle(image, (int(rect_original["x1"]), int(rect_original["y1"])),
                       2, color=(255, 255, 255), thickness=5, lineType=8)
            cv2.putText(image,
                        '{}{}'.format("original: ",
                                      (int(rect_original["x1"] * args.h_resize_coefficient),
                                       int(rect_original["y1"] * args.w_resize_coefficient))),
                        (int(rect_original["x1"]), int(rect_original["y1"]) - 30),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                        fontScale=1, thickness=1)
            cv2.putText(image,
                        '{}{}'.format("fixed: ", (int(rect_fixed["x1"]), int(rect_fixed["y1"]))),
                        (int(rect_original["x1"]), int(rect_original["y1"]) - 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                        fontScale=1, thickness=1)
        if not os.path.exists(comparison_save_folder):
            os.mkdir(comparison_save_folder)
        path_to_save = "{}/img{}.png".format(comparison_save_folder, "%06d" % frame_no)
        cv2.imwrite(path_to_save, image)


def fixed_coordinate_system(path_to_homography_dict):
    """
    Calculate and visualize new coordinat for original one from json
    """
    with open(args.path_to_original_coordinate, 'r') as path_to_detection:
        original_coordinates = json.load(path_to_detection)
        original_coordinates = {int(k): v for k, v in original_coordinates.items()}
    homography_dict = reformat_homography_dict(path_to_homography_dict)
    recalculated_coordinates = coordinates_recalculation(
        original_coordinates, homography_dict)
    with open("{}/recalculated_coordinates.json".format(args.save_folder), "w") as json_:
        json.dump(recalculated_coordinates, json_)
    capture = cv2.VideoCapture(args.path_to_video)
    visualize_comparison_original_with_fixed_coordinate(capture, original_coordinates,
                                                        recalculated_coordinates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments')
    parser.add_argument('--path_to_video', type=str, default="test_video/test_video.mp4")
    parser.add_argument('--experiment_folder', type=str, default="experiment")
    parser.add_argument('--experiment_name', type=str, default="test_video_processing")
    parser.add_argument('--resize_width', type=int, help="wights to resize image", default=400)
    parser.add_argument('--path_to_original_coordinate',
                        help="path to json with original coordinate",
                        default="test_video/original_coordinates.json")
    parser.add_argument('--none_H_processing', help="If True we use H_prev as H, False- do nothing",
                        default=True)
    parser.add_argument('--number_of_matching_points',
                        help="The minimum number of the matching points to find homography matrix",
                        default=4)
    parser.add_argument('--show_matches',
                        help="Show matches point or not", default=False)

    # heatmap visualization arguments
    parser.add_argument('--heatmap_visualization',
                        help="Getting heatmap visualization",
                        default=True)
    parser.add_argument('--heatmap_constant', help=" The constant for heatmap normalization",
                        default=1000)
    parser.add_argument('--number_of_vertical_lines',
                        help="The number of vertical lines in heatmap visualization",
                        default=5)
    parser.add_argument('--number_of_horizontal_lines',
                        help="The number of horizontal lines in heatmap visualization",
                        default=4)

    args = parser.parse_args()

    args.save_folder = args.experiment_folder + '/{}'.format(args.experiment_name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    args.save_folder = args.save_folder + '/{}'.format(os.path.split(args.path_to_video)[-1])
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    cap = cv2.VideoCapture(args.path_to_video)
    args.img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    args.img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result = get_json_with_homography_matrix(args.save_folder, cap)
    path_to_characteristic_dict = "{}/dict_with_homography_matrix.json".format(args.save_folder)

    # visualization
    if args.heatmap_visualization:
        cap = cv2.VideoCapture(args.path_to_video)
        template = make_template(result.shape)
        max_movement = heatmap_visualization_with_coordinates(
            path_to_characteristic_dict, template, cap)
        with open("{}/{}.txt".format(args.save_folder, "metrics_file"), "w") as txt_:
            txt_.write("Maximum movement during the entire video: {}".format(np.max(max_movement)))
            if np.max(max_movement) > INFINITY_COORDINATE_FLAG:
                txt_.write("There are some frames with undefined coordinates")
    ###
    # json with fixed coordinate system
    if args.path_to_original_coordinate:
        fixed_coordinate_system(path_to_characteristic_dict)
    ###
