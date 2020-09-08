"""
to visualize EvenVizion on grey background
"""
import argparse
import json
import os
from copy import deepcopy
import cv2
import imutils
import numpy as np


def homography_transformation(vector, matrix_H):
    """
    applying homography transformation with matrix H
    """
    new_vector = np.dot(matrix_H, vector)
    return new_vector / new_vector[-1]


def brightness_decrease(background_result, background):
    """
    decrease brightness of previous frame
    """
    hsv = cv2.cvtColor(background_result, cv2.COLOR_BGR2HSV)
    hsv_background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    v_value = np.array(deepcopy(hsv_background[..., 2]))
    hsv_background[..., 2] = np.where((v_value - 2) >= 254, 0, v_value - 2)
    for i, _ in enumerate(hsv):
        for j in range(0, len(hsv[i])):
            if hsv[i][j][2] >= 2:
                hsv[i][j][2] -= 2
            else:
                hsv[i][j] = [222, 12.35, 31.76]
    return hsv, hsv_background


def reformat_homography_dict(gomography_dict):
    """
    get matrix superposition for each frames
    """
    del gomography_dict["resize_info"]
    coner_1 = [0, 0, 1]
    x_corner = []
    y_corner = []
    matrix_H_next = None
    matrix_H_first = True
    matrix_H_superposition_dict = {}
    for frame_no, frame_H in gomography_dict.items():
        if frame_H["H"] is not None:
            if matrix_H_first:
                matrix_H_next = frame_H["H"]
                matrix_H_first = False
            else:
                matrix_H_next = np.dot(frame_H["H"], matrix_H_next)
        matrix_H_superposition_dict[frame_no] = matrix_H_next
        x_corner.append(homography_transformation(coner_1, matrix_H_next)[0])
        y_corner.append(homography_transformation(coner_1, matrix_H_next)[1])
    max_x = int(np.max(x_corner))
    min_x = int(np.min(x_corner))
    max_y = int(np.max(y_corner))
    min_y = int(np.min(y_corner))
    return matrix_H_superposition_dict, max_x, min_x, max_y, min_y


def frame_transformation(background, min_x, x_offset, min_y, y_offset, color_shape):
    """
    transform frame according matrix superposition
    """
    background_result = deepcopy(background)
    start_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset)
    end_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset + color_shape[0])
    background_result = cv2.line(background_result, start_point,
                                 end_point, color=[255, 255, 255], thickness=1)

    start_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset)
    end_point = (np.abs(min_x) + x_offset + color_shape[1], np.abs(min_y) + y_offset)
    background_result = cv2.line(background_result, start_point,
                                 end_point, color=[255, 255, 255], thickness=1)

    start_point = (np.abs(min_x) + x_offset + color_shape[1], np.abs(min_y) + y_offset)
    end_point = (np.abs(min_x) + x_offset + color_shape[1],
                 np.abs(min_y) + y_offset + color_shape[0])
    background_result = cv2.line(background_result, start_point,
                                 end_point, color=[255, 255, 255], thickness=1)

    start_point = (np.abs(min_x) + x_offset + color_shape[1],
                   np.abs(min_y) + y_offset + color_shape[0])
    end_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset + color_shape[0])
    background_result = cv2.line(background_result, start_point,
                                 end_point, color=[255, 255, 255], thickness=1)
    return background_result


def stabilization_view():
    """
    visualize components works
    """
    with open(args.path_to_homography_dict, "r") as curr_json:
        homography_dict = json.load(curr_json)
        width = homography_dict["resize_info"]["w"]
    cap = cv2.VideoCapture(args.path_to_original_video)
    _, image_template = cap.read()
    image_template = imutils.resize(image_template, width=width)
    homography_superposition_dict, max_x, min_x, max_y, min_y = \
        reformat_homography_dict(homography_dict)
    panorama_shape = [int(np.abs(min_y) + max_y + image_template.shape[0] + 10),
                      int(np.abs(min_x) + max_x + image_template.shape[1] + 10)]

    background = np.full((panorama_shape[0], panorama_shape[1], 3), 0, dtype=np.uint8)
    background[np.abs(min_y):np.abs(min_y) + image_template.shape[0],
               np.abs(min_x):np.abs(min_x) + image_template.shape[1]] = image_template
    color_shape = [image_template.shape[0], image_template.shape[1]]
    for (frame_no, frame_H) in homography_superposition_dict.items():
        _, image_new = cap.read()
        image_new = imutils.resize(image_new, width=width)
        transform_vector = (np.dot(frame_H, [0, 0, 1]))
        x_offset = int(transform_vector[0] / transform_vector[2])
        y_offset = int(transform_vector[1] / transform_vector[2])
        background[np.abs(min_y) + y_offset:np.abs(min_y) + y_offset + color_shape[0],
                   np.abs(min_x) + x_offset:np.abs(min_x) + x_offset + color_shape[1]] = image_new
        background_result = frame_transformation(background, min_x,
                                                 x_offset, min_y, y_offset, color_shape)
        hsv, hsv_background = brightness_decrease(background_result,
                                                  background)
        background_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        background = cv2.cvtColor(hsv_background, cv2.COLOR_HSV2BGR)
        cv2.imwrite("{}/img{}.png".format(args.save_folder,
                                          "%06d" % int(frame_no)), background_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments without using Sacred library ')
    parser.add_argument('--path_to_homography_dict', help="path to homography dict",
                        default="experiment/test_video_processing/dict_with_homography_matrix.json")
    parser.add_argument('--path_to_original_video',
                        default="test_video/test_video.mp4")
    parser.add_argument('--experiment_name',
                        default="visualize_camera_stabilization")
    parser.add_argument('--experiment_folder', help="folder to save experiment result",
                        default="experiment/test_video_processing")
    args = parser.parse_args()
    args.save_folder = args.experiment_folder + "/" + args.experiment_name
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    stabilization_view()
