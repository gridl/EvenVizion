"""
stitch two visualization (compare real video and EvenVizion)
"""
import argparse
import logging
import os
import cv2
import imutils
import numpy as np

logging.basicConfig(level=logging.INFO)


def create_border(image_b):
    """
    create image border
    """
    pt1 = (0, 0)
    pt2 = (image_b.shape[1] - 1, image_b.shape[0] - 1)
    image_b = cv2.rectangle(image_b, pt1, pt2, color=[0, 0, 248], thickness=1, lineType=8, shift=0)
    return image_b


def image_design(image_path, text, height=300):
    """
    put text on frame
    """
    if text == "EvenVizion":
        image = cv2.imread(image_path)
    elif text == "Original":
        image = image_path
    else:
        raise ValueError("problem with type in image_design")
    image = imutils.resize(image, height=height)
    image = cv2.putText(image, text, (10, 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                        fontScale=1, thickness=1)
    return image


def get_pictures(path_to_folder):
    """
    get Even Vizion pictures
    """
    path_to_folder_with_pictures_ = os.listdir(path_to_folder)
    pictures_original_pathes = [path_to_folder + "/" + f for f in path_to_folder_with_pictures_ if
                                (f.endswith('.png') or f.endswith('.jpg'))]
    pictures_original_pathes.sort()
    return pictures_original_pathes


def draw_video():
    """
    create two pictures comparison
    """
    cap = cv2.VideoCapture(args.path_to_original_video)
    _, _ = cap.read()
    pictures_evenvizion_pathes = get_pictures(args.path_to_evenvizion_result_frames)
    i = 1
    for evenvizion_picture_path in pictures_evenvizion_pathes:
        i += 1
        _, original_picture_path = cap.read()
        image_a = image_design(original_picture_path, "Original")
        image_b = image_design(evenvizion_picture_path, "EvenVizion")
        logo = cv2.imread(args.path_to_logo)
        logo = imutils.resize(logo, 64)
        image_b[-logo.shape[0]:, -logo.shape[1]:, :] = logo
        image_b = create_border(image_b)
        result = np.concatenate((image_a, image_b), axis=1)
        cv2.imwrite("{}/{}.png".format(args.save_folder, "%06d" % i), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='custom arguments')
    parser.add_argument('--path_to_original_video',
                        default="test_video/test_video.mp4")
    parser.add_argument('--path_to_evenvizion_result_frames',
                        help="path to folder with frames after evenvizion_visualization.py",
                        default="experiment/test_video_processing/visualize camera stabilization")
    parser.add_argument('--path_to_logo',
                        default="oxagile_logo.png")
    parser.add_argument('--experiment_folder', help="folder to save experiment result",
                        default="experiment/test_video_processing")
    parser.add_argument('--experiment_name', type=str, default="original_video_with_EvenVizion")
    args = parser.parse_args()
    args.save_folder = args.experiment_folder + '/{}'.format(args.experiment_name)
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    draw_video()
