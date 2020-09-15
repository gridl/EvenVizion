"""
EvenVizion library.
https://github.com/RnD-Oxagile/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf

This is licensed under an MIT license. See the README.md file
for more information.

This is an example of getting homography matrix for each frame, getting heatmap visualization.
If you specify --path_to_original_coordinate,
original coordinates will be recalculated to a fixed coordinate system.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(main_dir)

from evenvizion.processing.fixed_coordinate_system import from_original_to_fix
from evenvizion.processing.utils import read_homography_dict, superposition_dict, \
    are_infinity_coordinates, read_json_with_coordinates
from evenvizion.processing.video_processing import get_homography_dict
from evenvizion.visualization.processing_visualization import \
    heatmap_video_processing, comparison_original_with_fixed_coordinate_video_processing



def visualize_heatmap(save_folder, path_to_video, path_to_homography_dict):
    """ To create heatmap visualization.

    Parameters
    ----------
    save_folder: str
        The path to save pictures with heatmap visualization.

    path_to_video: str
        The path to the analyzed video.

    path_to_homography_dict: str
        The path to the json file with homography matrix,
        which describes movement between consistent frames on video.

    """
    cap = cv2.VideoCapture(path_to_video)
    homography_matrices, _ = read_homography_dict(path_to_homography_dict)
    reformat_homography_dict = superposition_dict(homography_matrices)
    heatmap_save_folder = "{}/heatmap_visualization".format(save_folder)
    if not os.path.exists(heatmap_save_folder):
        os.mkdir(heatmap_save_folder)
    max_movement = heatmap_video_processing(
        reformat_homography_dict, cap, heatmap_save_folder)
    with open("{}/{}.txt".format(args.save_folder, "metrics_file"), "w") as txt_:
        txt_.write("Maximum movement during the entire video: {}".format(np.max(max_movement)))
        if are_infinity_coordinates(max_movement):
            txt_.write("There are some frames with undefined coordinates")


def get_fixed_coordinate(path_to_video, path_to_homography_dict, original_shape):
    """ To recalculate and visualize original coordinates to fixed.

    Parameters
    ----------
    path_to_video: str
        The path to the analyzed video.

    path_to_homography_dict: str
        The path to the json file with homography matrix,
        which describes movement between consistent frames on video.

    original_shape: list, or np.array
        An original frame shape, at the first place - height, the second - width.
        example:  [224,400]

    """
    dict_with_homography_matrix, resize_info = read_homography_dict(path_to_homography_dict)
    reformat_homography_dict = superposition_dict(dict_with_homography_matrix)
    original_coordinates = read_json_with_coordinates(args.path_to_original_coordinate)
    recalculated_coordinates = from_original_to_fix(original_coordinates,
                                                    reformat_homography_dict, original_shape,
                                                    [resize_info["h"], resize_info["w"]])
    cap = cv2.VideoCapture(path_to_video)
    comparison_save_folder = "{}/fixed_coordinate_system_visualization".format(args.save_folder)
    comparison_original_with_fixed_coordinate_video_processing(cap, original_coordinates,
                                                               recalculated_coordinates,
                                                               comparison_save_folder,
                                                               resize_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments')
    parser.add_argument('--path_to_video', type=str, default="test_video/test_video.mp4")
    parser.add_argument('--experiment_name', type=str, default="test_video_processing")
    parser.add_argument('--resize_width', type=int, help="wights to resize image", default=400)
    parser.add_argument('--path_to_original_coordinate',
                        help="path to json with original coordinate",
                        default="test_video/original_coordinates.json")
    parser.add_argument('--none_H_processing', help="If True we use H_prev as H, False- do nothing",
                        default=True)
    parser.add_argument('--heatmap_visualization',
                        help="Getting heatmap visualization",
                        default=True)
    parser.add_argument('--show_matching_visualization',
                        help="Getting matching visualization",
                        default=True)

    args = parser.parse_args()
    args.experiment_folder = os.getcwd()
    args.save_folder = args.experiment_folder + "/" + args.experiment_name
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    args.save_folder = args.save_folder + '/{}'.format(os.path.split(args.path_to_video)[-1].split(".")[0])
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    path_to_save_matching = None
    if args.show_matching_visualization:
        path_to_save_matching = os.path.join(args.save_folder, "matching_visualization")
        if not os.path.exists(path_to_save_matching):
            os.makedirs(path_to_save_matching)

    cap = cv2.VideoCapture(args.path_to_video)
    original_img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_shape = [original_img_h, original_img_w]
    result = get_homography_dict(cap, matching_path=path_to_save_matching,
                                 none_H_processing=args.none_H_processing)

    with open("{}/dict_with_homography_matrix.json".format(args.save_folder), "w") as json_:
        json.dump(result, json_)
    path_to_homography_dict = "{}/dict_with_homography_matrix.json".format(args.save_folder)

    if args.heatmap_visualization:
        visualize_heatmap(args.save_folder, args.path_to_video, path_to_homography_dict)
    if args.path_to_original_coordinate:
        get_fixed_coordinate(args.path_to_video, path_to_homography_dict, original_shape)
