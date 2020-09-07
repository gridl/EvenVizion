"""
to compare EvenVizion with original video
"""
import argparse
import os

from EvenVizion.visualization.stabilization import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments without using Sacred library ')
    parser.add_argument('--path_to_homography_dict', help="path to homography dict",
                        default="experiment/test_video_processing/dict_with_homography_matrix.json")
    parser.add_argument('--path_to_video',
                        default="test_video/test_video.mp4")
    parser.add_argument('--experiment_name',
                        default="visualize_camera_stabilization")
    parser.add_argument('--experiment_folder', help="folder to save experiment result")
    args = parser.parse_args()
    args.experiment_folder = os.getcwd()
    args.save_folder = args.experiment_folder + "/" + args.experiment_name
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    homography_dict, resize_info = read_homography_dict(args.path_to_homography_dict)
    reformat_homography_dict = superposition_dict(homography_dict)
    cap = cv2.VideoCapture(args.path_to_video)

    create_video_comparison(args.path_to_video, args.save_folder, reformat_homography_dict, resize_info["w"])
