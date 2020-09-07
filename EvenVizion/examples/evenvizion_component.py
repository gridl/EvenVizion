import argparse
import json
import logging
import os

import cv2
import numpy as np

from EvenVizion.processing.fixed_coordinate_system import *
from EvenVizion.processing.video_processing import *
from EvenVizion.visualization.processing_visualization import *

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
                        default=False)
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

    cap = cv2.VideoCapture(args.path_to_video)
    args.img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    args.img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result = get_homography_dict(cap)

    with open("{}/dict_with_homography_matrix.json".format(args.save_folder), "w") as json_:
        json.dump(result, json_)
    path_to_characteristic_dict = "{}/dict_with_homography_matrix.json".format(args.save_folder)

    # visualization
    if args.heatmap_visualization:
        logging.info("Start heatmap visualization")
        cap = cv2.VideoCapture(args.path_to_video)
        dict_with_homography_matrix, resize_info = read_homography_dict(path_to_characteristic_dict)
        reformat_homography_dict = superposition_dict(dict_with_homography_matrix)
        heatmap_save_folder = "{}/heatmap_visualization".format(args.save_folder)
        if not os.path.exists(heatmap_save_folder):
            os.mkdir(heatmap_save_folder)
        max_movement = heatmap_video_processing(
            reformat_homography_dict, cap, heatmap_save_folder, heatmap_constant=1000)
        with open("{}/{}.txt".format(args.save_folder, "metrics_file"), "w") as txt_:
            txt_.write("Maximum movement during the entire video: {}".format(np.max(max_movement)))
            if np.max(max_movement) > INFINITY_COORDINATE_FLAG:
                txt_.write("There are some frames with undefined coordinates")
    dict_with_homography_matrix, resize_info = read_homography_dict(path_to_characteristic_dict)
    reformat_homography_dict = superposition_dict(dict_with_homography_matrix)
    original_coordinates, recalculated_coordinates = fixed_coordinate_system(reformat_homography_dict,
                                                                             args.path_to_original_coordinate,
                                                                             [args.img_h, args.img_w],
                                                                             [resize_info["h"], resize_info["w"]])
    cap = cv2.VideoCapture(args.path_to_video)
    comparison_save_folder = "{}/fixed_coordinate_system_visualization".format(args.save_folder)
    comparison_original_with_fixed_coordinate_video_processing(cap, original_coordinates,
                                                               recalculated_coordinates, comparison_save_folder, resize_info)
