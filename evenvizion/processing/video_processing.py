"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

Function to get homography dict for video.
"""
import logging
import os

import cv2
import imutils

logging.basicConfig(level=logging.DEBUG)

from evenvizion.processing.frame_processing import FrameProcessing
from evenvizion.processing.matching import NoMatchesException
from evenvizion.processing.utils import compute_homography, \
    HomographyException, matrix_superposition
from evenvizion.visualization.processing_visualization import draw_matches


def get_homography_dict(capture, resize_width=400, matching_path=None, none_H_processing=True):
    """ Returns a dictionary of homography matrices for each frame of the video.

    Parameters
    ----------

    capture: cv2.VideoCapture
        The analyzed video.

    resize_width: int, optional
        To speed up the performance of the script, pictures will be resized to this width.

    matching_path: str, optional
        If is not None, create matching visualization in this folder.


    none_H_processing: bool, optional
     There are some cases where Homography matrix can't be calculated,
            so you need to choose which script you want to run in this case.
            If set True H = H in the previous step,
            False - H = [1,0,0][0,1,0][0,0,1], it means there is no transformation on this frame.

    Returns
    ----------

    homography_dict: dict
     format: {frame_no: {"H": [[.,.,.],[.,.,.],[.,.,.]]}
              ...
              frame_no: {"H": [[.,.,.],[.,.,.],[.,.,.]]
              resize_info:{"h": int, "w": int}}.
    """
    success, result_image = capture.read()
    homography_dict = {}
    if not success:
        raise ValueError("Problem with video! Can't read first frame")
    result_image = imutils.resize(result_image, width=resize_width)
    i = 1
    matrix_H_superposition = None
    matrix_H_prev = None
    matrix_H_first = True
    while success:
        i += 1
        frame_processing_a = FrameProcessing(result_image)
        success, image_b = capture.read()
        if not success:
            continue
        image_b = imutils.resize(image_b, width=resize_width)
        frame_processing_b = FrameProcessing(image_b)
        try:
            key_points_a, key_points_b = \
                frame_processing_b.concatenate_all_features_types(frame_processing_a)
            if matching_path:
                matching_visualization = \
                    draw_matches(frame_processing_a.frame, frame_processing_b.frame, key_points_a, key_points_b)
                cv2.imwrite(os.path.join(matching_path, "matching_vis_{}.png".format(i)), matching_visualization)

            matrix_H = compute_homography(key_points_a, key_points_b, matrix_H_superposition)
        except NoMatchesException:
            logging.info(NoMatchesException)
            matrix_H = None
        except HomographyException:
            logging.info(HomographyException)
            matrix_H = None

        result_image = image_b

        # matrix_H is None case processing
        if matrix_H is None:
            if none_H_processing:
                matrix_H = matrix_H_prev
            else:
                homography_dict[i] = {"H": None}
        ###

        homography_dict[i] = {"H": matrix_H.tolist()}
        matrix_H_superposition = \
            matrix_superposition(matrix_H, matrix_H_superposition, matrix_H_first)
        matrix_H_prev = matrix_H
        matrix_H_first = False
    homography_dict["resize_info"] = \
        {"h": result_image.shape[0], "w": result_image.shape[1]}
    return homography_dict
