"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

All the functions used to recalculate coordinates into different coordinate system are located here.
"""

from copy import deepcopy
import numpy as np

from evenvizion.processing.utils import inverse_homography_transformation, homography_transformation


def from_original_to_fix(original_coordinates, homography_dict,
                         original_image_shape, resize_image_shape):
    """ To recalculate original coordinates into a fixed coordinate system.

    Parameters
    ----------
    original_coordinates: dict
        The original coordinates of points.
            example: {frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}],
                     ...,
                     frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}]}

    homography_dict: dict
        A dictionary with homography matrices for each frame of video.
            format: {frame_no: {"H": [[.,.,.],[.,.,.],[.,.,.]]}
                    ...
                    frame_no: {"H": [[.,.,.],[.,.,.],[.,.,.]]
                    resize_info: {"h": int, "w": int}}

    original_image_shape: list, or np.array
        An original frame shape, at the first place - height, the second - width.
            example: [224,400].

    resize_image_shape: list, or np.array
        A shape of the frame on which homography matrix was received.
        At the first place - height, the second - width.
            example: [224,400].

    Returns
    ----------
    fixed_coordinates: dict
        A dictionary with original_coordinates in a fixed coordinate system.
            format: {frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}],
                    ...,
                    frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}]}

    """
    original_h, original_w = original_image_shape
    resize_h, resize_w = resize_image_shape
    fixed_coordinates = {}
    h_resize_coefficient = int(resize_h) / original_h
    w_resize_coefficient = int(resize_w) / original_w
    for frame_no, frame_info in original_coordinates.items():
        fixed_coordinates[frame_no] = []
        for rect in frame_info:
            new_rect = deepcopy(rect)
            [new_rect["x1"], new_rect["y1"]] = np.around(homography_transformation(
                [w_resize_coefficient * rect["x1"], h_resize_coefficient * rect["y1"]],
                homography_dict[frame_no]), decimals=2)
            fixed_coordinates[frame_no].append(new_rect)
    return fixed_coordinates


def from_fix_to_original(fix_coordinates, homography_dict,
                         original_image_shape, resize_image_shape):
    """ To recalculate fixed coordinates into a coordinate system relative to the frame.

    Parameters
    ----------
    fix_coordinates: dict
        The original coordinates of points in a fixed system coordinate.
            format: {frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}],
                    ...,
                    frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}]}

    homography_dict: dict
       A dictionary with homography matrices for each frame of video.
            format: {frame_no: {"H": [[.,.,.],[.,.,.],[.,.,.]]}
                    ...
                    frame_no: {"H": [[.,.,.],[.,.,.],[.,.,.]]
                    resize_info: {"h": int, "w": int}}

    original_image_shape: list, or np.array
        An original frame shape, at the first place - height, the second - width.
            example: [224,400].

    resize_image_shape: list, or np.array
        A shape of frame on which homography matrix was received.
        At the first place - height, the second - width.
            example: [224,400].

    Returns
    ----------
    original_coordinates: dict
        A dictionary with original_coordinates in a fixed system coordinate.
            format: {frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}],
                    ...,
                    frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}]}
    """

    original_h, original_w = original_image_shape
    resize_h, resize_w = resize_image_shape
    original_coordinates = {}
    h_resize_coefficient = original_h / resize_h
    w_resize_coefficient = original_w / resize_w
    for frame_no, frame_info in fix_coordinates.items():
        original_coordinates[frame_no] = []
        for rect in frame_info:
            new_rect = deepcopy(rect)
            [new_rect["x1"], new_rect["y1"]] = np.around(inverse_homography_transformation(
                [w_resize_coefficient * rect["x1"], h_resize_coefficient * rect["y1"]],
                homography_dict[frame_no]), decimals=2)
            original_coordinates[frame_no].append(new_rect)
    return original_coordinates
