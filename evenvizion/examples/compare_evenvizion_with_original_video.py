"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

This is an example of the comparison of visualizations between the original video and stabilized video.
"""

import argparse
import os
import cv2
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(main_dir)


from evenvizion.visualization.stabilization import create_video_comparison
from evenvizion.processing.utils import superposition_dict, read_homography_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments without using Sacred library')
    parser.add_argument('--path_to_homography_dict', help="path to homography dict",
                        default="test_video_processing/test_video/dict_with_homography_matrix.json")
    parser.add_argument('--path_to_video',
                        default="test_video/test_video.mp4")
    parser.add_argument('--experiment_name', help="folder to save experiment result", default="test_video_processing")
    args = parser.parse_args()
    args.experiment_folder = os.getcwd()
    args.save_folder = args.experiment_folder + "/" + args.experiment_name
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    args.save_folder = args.save_folder + '/{}'.format(os.path.split(args.path_to_video)[-1].split(".")[0])
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    args.save_folder = args.save_folder + "/" + "visualize_camera_stabilization"
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    homography_dict, resize_info = read_homography_dict(args.path_to_homography_dict)
    reformat_homography_dict = superposition_dict(homography_dict)
    cap = cv2.VideoCapture(args.path_to_video)
    create_video_comparison(args.path_to_video, args.save_folder,
                            reformat_homography_dict, resize_info["w"])
