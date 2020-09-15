"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

All visualization function using to get stabilization view.
"""
from copy import deepcopy

import cv2
import imutils
import numpy as np

from evenvizion.processing.utils import homography_transformation


def decrease_brightness(background_result, background):
    """ To decrease brightness of previous frame.

        Parameters
        ----------
        background_result: np.array
            An image that is being drawn.

        background: np.array
            The original coordinates of a point.
                format: {"x1": float, "y1": float}

       """
    gray_background = cv2.cvtColor(background_result, cv2.COLOR_BGR2HSV)
    black_background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    v_value = np.array(deepcopy(black_background[..., 2]))
    black_background[..., 2] = np.where((v_value - 2) >= 254, 0, v_value - 2)
    for i, _ in enumerate(gray_background):
        for j in range(0, len(gray_background[i])):
            if gray_background[i][j][2] >= 2:
                gray_background[i][j][2] -= 2
            else:
                gray_background[i][j] = [222, 12.35, 31.76]
    return gray_background, black_background


def change_frame_location(black_background, min_x, x_offset, min_y, y_offset, image_shape):
    """ To draw white border for a video frame.

    Parameters
    ----------
    black_background: np.array
        An image that is being drawn.

    min_x: float
        x coordinate of the upper left corner.

    min_y: float
        y coordinate of the upper left corner.

    x_offset: float
        displacement x coordinate of the upper left corner.

    y_offset: float
        displacement y coordinate of the upper left corner.

    image_shape:
        An original frame shape, at the first place - height, the second - width

    Returns
    ----------
    np.array
        An image with white border for video frame.
    """
    color_background = deepcopy(black_background)
    start_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset)
    end_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset + image_shape[0])
    color_background = cv2.line(color_background, start_point,
                                end_point, color=[255, 255, 255], thickness=1)

    start_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset)
    end_point = (np.abs(min_x) + x_offset + image_shape[1], np.abs(min_y) + y_offset)
    color_background = cv2.line(color_background, start_point,
                                end_point, color=[255, 255, 255], thickness=1)

    start_point = (np.abs(min_x) + x_offset + image_shape[1], np.abs(min_y) + y_offset)
    end_point = (np.abs(min_x) + x_offset + image_shape[1],
                 np.abs(min_y) + y_offset + image_shape[0])
    color_background = cv2.line(color_background, start_point,
                                end_point, color=[255, 255, 255], thickness=1)

    start_point = (np.abs(min_x) + x_offset + image_shape[1],
                   np.abs(min_y) + y_offset + image_shape[0])
    end_point = (np.abs(min_x) + x_offset, np.abs(min_y) + y_offset + image_shape[0])
    color_background = cv2.line(color_background, start_point,
                                end_point, color=[255, 255, 255], thickness=1)
    return color_background


def get_reference_system(homography_dict):
    """ Return dictionary with min and max coordinate in a fixed coordinate system.

    Parameters
    ----------
    homography_dict: dict
        A dictionary with matrix superposition for each frame.
        If you want to analyze displacement only from the previous frame,
        you can specify homography dict without superposition.
            format: {frame_no: [],
                          ...
                      frame_no: []}
    Returns
    ----------
    dict
        The dictionary with min and max coordinate in a fixed coordinate system.
    """
    corner_1 = [0, 0, 1]
    x_corner = []
    y_corner = []
    for _, H in homography_dict.items():
        x_corner.append(homography_transformation(corner_1, H)[0])
        y_corner.append(homography_transformation(corner_1, H)[1])
    corner_dict = {"max_x": int(np.max(x_corner)), "min_x": int(np.min(x_corner)),
                   "max_y": int(np.max(y_corner)),
                   "min_y": int(np.min(y_corner))}
    return corner_dict


def stabilize_view(original_frame, matrix_H, background, corner_dict, width):
    """ To draw a frame position in a fixed coordinate system.

    Parameters
    ----------
    original_frame: np.array
        The frame which position we want to find.

    matrix_H: np.array
        Homography matrix describing the movement of this frame.

    background: np.array
        Background of a fixed coordinate system plane.

    corner_dict: dict
        The dictionary with min and max coordinate in a fixed coordinate system.

    width: int
        Frame width on which homography matrix was received.

    Returns
    ----------
    stabilize_frame: np.array
        The image with frame position on a fixed coordinate system plane.

    background: np.array
        Background of a fixed coordinate system plane with the original frame.
    """
    original_frame = imutils.resize(original_frame, width=width)
    color_shape = [original_frame.shape[0], original_frame.shape[1]]
    transform_vector = (np.dot(matrix_H, [0, 0, 1]))
    x_offset = int(transform_vector[0] / transform_vector[2])
    y_offset = int(transform_vector[1] / transform_vector[2])
    background[np.abs(
        corner_dict["min_y"]) + y_offset:np.abs(corner_dict["min_y"]) + y_offset + color_shape[0],
               np.abs(corner_dict["min_x"]) + x_offset:np.abs(
                   corner_dict["min_x"]) + x_offset + color_shape[1]] = original_frame
    stabilize_frame = change_frame_location(background, corner_dict["min_x"],
                                            x_offset, corner_dict["min_y"], y_offset, color_shape)
    hsv, hsv_background = decrease_brightness(stabilize_frame,
                                              background)
    stabilize_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    background = cv2.cvtColor(hsv_background, cv2.COLOR_HSV2BGR)
    return stabilize_frame, background


def create_border(image_b):
    """ To create border for a fixed coordinate system plane

    Parameters
    ----------
    image_b: np.array
        Image that is being drawn.

    Returns
    ----------
    np.array:
        The image with border.
    """
    pt1 = (0, 0)
    pt2 = (image_b.shape[1] - 1, image_b.shape[0] - 1)
    image_b = cv2.rectangle(image_b, pt1, pt2, color=[0, 0, 248], thickness=1, lineType=8, shift=0)
    return image_b


def append_text_to_image(image, text, height=300):
    """To put text on a frame

    Parameters
    ----------
    image: np.array
        Image that is being drawn.

    text: str
        The text you want to write on the frame.

    height: int, optional
        If you want a resized image specify height.

    Returns
    ----------
    np.array:
        The image with text.
    """
    image = imutils.resize(image, height=height)
    image = cv2.putText(image, text, (10, 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                        fontScale=1, thickness=1)
    return image


def initialize_background(original_frame, width, corner_dict):
    """To initialize image with fixed coordinate system plane.

    Parameters
    ----------
    corner_dict: dict
         The dictionary with min and max coordinate in fixed coordinate system.

    original_frame: np.array
        The frame from video.

    width: int
        Frame width on which homography matrix was received.

    Returns
    ----------
    np.array:
        The image with a  fixed coordinate system plane.
    """

    image_template = imutils.resize(original_frame, width=width)
    panorama_shape = [int(np.abs(corner_dict["min_y"]) +
                          corner_dict["max_y"] + image_template.shape[0] + 10),
                      int(np.abs(corner_dict["min_x"]) +
                          corner_dict["max_x"] + image_template.shape[1] + 10)]
    background = np.full((panorama_shape[0], panorama_shape[1], 3), 0, dtype=np.uint8)
    background[np.abs(corner_dict["min_y"]):np.abs(corner_dict["min_y"]) + image_template.shape[0],
               np.abs(corner_dict["min_x"]):np.abs(corner_dict["min_x"]) +
               image_template.shape[1]] = image_template
    return background


def create_video_comparison(path_to_video, save_folder, homography_dict, width):
    """To save comparison of a real frame from video with stabilization frame.

    Parameters
    ----------
    path_to_video: str
        The path to the analyzed video.

    save_folder: str
        The path to save pictures with heatmap visualization.

    homography_dict: dict
        Dict with matrix superposition for each frame.
        If you want to analyze displacement analyze only from the previous frame,
        you can specify homography dict without superposition.
            format: {frame_no: [],
                          ...
                      frame_no: []}

    width: int
        Frame width on which homography matrix was received.
    """
    cap = cv2.VideoCapture(path_to_video)
    success, original_frame = cap.read()
    i = 1
    corner_dict = get_reference_system(homography_dict)
    background = initialize_background(original_frame, width, corner_dict)
    while success:
        image_a = append_text_to_image(original_frame, "Original")
        stabilize_frame, background = stabilize_view(original_frame, homography_dict[i], background,
                                                     corner_dict, width)
        image_b = append_text_to_image(stabilize_frame, "EvenVizion")
        image_b = create_border(image_b)
        result = np.concatenate((image_a, image_b), axis=1)
        cv2.imwrite("{}/{}.png".format(save_folder, "%06d" % i), result)
        i += 1
        success, original_frame = cap.read()
        if not success:
            break
