"""
to visualize EvenVizion on grey background
"""
from copy import deepcopy

import imutils

from EvenVizion.processing.utils import *

logging.basicConfig(level=logging.INFO)


def decrease_brightness(background_result, background):
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


def transform_frame(background, min_x, x_offset, min_y, y_offset, color_shape):
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


def get_reference_system(homography_dict):
    """
    get the сentral location of the frame
    """
    corner_1 = [0, 0, 1]
    x_corner = []
    y_corner = []
    for _, H in homography_dict.items():
        x_corner.append(homography_transformation(corner_1, H)[0])
        y_corner.append(homography_transformation(corner_1, H)[1])
    corner_dict = {"max_x": int(np.max(x_corner)), "min_x": int(np.min(x_corner)), "max_y": int(np.max(y_corner)),
                   "min_y": int(np.min(y_corner))}
    return corner_dict


def stabilize_view(original_frame, H, width, background, corner_dict):
    """
    visualize components works
    """
    original_frame = imutils.resize(original_frame, width=width)
    color_shape = [original_frame.shape[0], original_frame.shape[1]]
    transform_vector = (np.dot(H, [0, 0, 1]))
    x_offset = int(transform_vector[0] / transform_vector[2])
    y_offset = int(transform_vector[1] / transform_vector[2])
    background[np.abs(corner_dict["min_y"]) + y_offset:np.abs(corner_dict["min_y"]) + y_offset + color_shape[0],
    np.abs(corner_dict["min_x"]) + x_offset:np.abs(corner_dict["min_x"]) + x_offset + color_shape[1]] = original_frame
    stabilize_frame = transform_frame(background, corner_dict["min_x"],
                                      x_offset, corner_dict["min_y"], y_offset, color_shape)
    hsv, hsv_background = decrease_brightness(stabilize_frame,
                                              background)
    stabilize_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    background = cv2.cvtColor(hsv_background, cv2.COLOR_HSV2BGR)
    return stabilize_frame, background


def create_border(image_b):
    """
    create image border
    """
    pt1 = (0, 0)
    pt2 = (image_b.shape[1] - 1, image_b.shape[0] - 1)
    image_b = cv2.rectangle(image_b, pt1, pt2, color=[0, 0, 248], thickness=1, lineType=8, shift=0)
    return image_b


def append_text_to_image(image, text, height=300):
    """
    put text on frame
    """
    image = imutils.resize(image, height=height)
    image = cv2.putText(image, text, (10, 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                        fontScale=1, thickness=1)
    return image


def initialize_background(homography_dict, original_frame, width):
    image_template = imutils.resize(original_frame, width=width)
    corner_dict = get_reference_system(homography_dict)
    panorama_shape = [int(np.abs(corner_dict["min_y"]) + corner_dict["max_y"] + image_template.shape[0] + 10),
                      int(np.abs(corner_dict["min_x"]) + corner_dict["max_x"] + image_template.shape[1] + 10)]
    background = np.full((panorama_shape[0], panorama_shape[1], 3), 0, dtype=np.uint8)
    background[np.abs(corner_dict["min_y"]):np.abs(corner_dict["min_y"]) + image_template.shape[0],
    np.abs(corner_dict["min_x"]):np.abs(corner_dict["min_x"]) + image_template.shape[1]] = image_template
    return background


def create_video_comparison(path_to_video, save_folder, homography_dict, width):
    """
    create two pictures comparison
    """
    cap = cv2.VideoCapture(path_to_video)
    success, original_frame = cap.read()
    i = 1
    corner_dict = get_reference_system(homography_dict)
    background = initialize_background(homography_dict, original_frame, width)
    while success:
        image_a = append_text_to_image(original_frame, "Original")
        success, original_frame = cap.read()
        stabilize_frame, background = stabilize_view(original_frame, homography_dict[i], width, background,
                                                     corner_dict)
        image_b = append_text_to_image(stabilize_frame, "EvenVizion")
        image_b = create_border(image_b)
        result = np.concatenate((image_a, image_b), axis=1)
        cv2.imwrite("{}/{}.png".format(save_folder, "%06d" % i), result)
        i += 1
        success, original_frame  = cap.read()
