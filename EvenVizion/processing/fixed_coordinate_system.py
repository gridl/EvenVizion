from copy import deepcopy

from EvenVizion.processing.frame_processing import *


def from_original_to_fix(original_coordinates, homography_dict, original_image_shape, resize_image_shape):
    """
    Recalculate original coordinates into fixed coordinate system
    """
    original_h, original_w = original_image_shape
    resize_h, resize_w = resize_image_shape
    recalculated_detection_coordinates = {}
    h_resize_coefficient = int(resize_h) / original_h
    w_resize_coefficient = int(resize_w) / original_w
    for frame_no, frame_info in original_coordinates.items():
        recalculated_detection_coordinates[frame_no] = []
        for rect in frame_info:
            new_rect = deepcopy(rect)
            [new_rect["x1"], new_rect["y1"]] = np.around(homography_transformation(
                [w_resize_coefficient * rect["x1"], h_resize_coefficient * rect["y1"]],
                homography_dict[frame_no]), decimals=2)
            recalculated_detection_coordinates[frame_no].append(new_rect)
    return recalculated_detection_coordinates


def fixed_coordinate_system(homography_dict, path_to_original_coordinate, original_image_shape, resize_image_shape):
    """
    Calculate and visualize new coordinat for original one from json
    """
    with open(path_to_original_coordinate, 'r') as path_to_detection:
        original_coordinates = json.load(path_to_detection)
        original_coordinates = {int(k): v for k, v in original_coordinates.items()}
    recalculated_coordinates = from_original_to_fix(
        original_coordinates, homography_dict, original_image_shape, resize_image_shape)
    return original_coordinates, recalculated_coordinates


def from_fix_to_original(fix_coordinates, homography_dict, original_image_shape, resize_image_shape):
    """
    Recalculate original coordinates to fixed coordinate system
    """
    original_h, original_w = original_image_shape
    resize_h, resize_w = resize_image_shape
    recalculated_detection_coordinates = {}
    h_resize_coefficient = original_h / resize_h
    w_resize_coefficient = original_w / resize_w
    for frame_no, frame_info in fix_coordinates.items():
        recalculated_detection_coordinates[frame_no] = []
        for rect in frame_info:
            new_rect = deepcopy(rect)
            [new_rect["x1"], new_rect["y1"]] = np.around(inverse_homography_transformation(
                [w_resize_coefficient * rect["x1"], h_resize_coefficient * rect["y1"]],
                homography_dict[frame_no]), decimals=2)
            recalculated_detection_coordinates[frame_no].append(new_rect)
    return recalculated_detection_coordinates
