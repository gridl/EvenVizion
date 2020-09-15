"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

All the visualization functions used during finding a fixed coordinate system.
"""
import os

import cv2
import imutils
import numpy as np

from evenvizion.processing.constants import HEATMAP_CONSTANT
from evenvizion.processing.utils import are_infinity_coordinates, homography_transformation


def draw_matches(image_a, image_b, pts_a, pts_b):
    """ To visualize matching points between two images.

    Parameters
    ----------
    image_a: np.array
     The first image.

    image_b: np.array
     The second image.jr

    pts_a: np.array
            An array of x and y coordinates of each key point
            which matches with points from pts_b.

    pts_b: np.array
         An array of x and y coordinates of each key point
         which matches with points from pts_a.

    Returns
    ----------
    np.array
        Matching points visualization.
    """
    (h_a, w_a) = image_a.shape[:2]
    (h_b, w_b) = image_b.shape[:2]
    visualization = np.zeros((max(h_a, h_b), w_a + w_b, 3), dtype="uint8")
    visualization[0:h_a, 0:w_a] = image_a
    visualization[0:h_b, w_a:] = image_b
    counter = 0
    for (a_point, b_point) in zip(pts_a, pts_b):
        counter += 1
        pt_a = (int(a_point[0]), int(a_point[1]))
        pt_b = (int(b_point[0]) + w_a, int(b_point[1]))
        cv2.line(visualization, pt_a, pt_b, (0, 255, 0), 1)
    return visualization


def find_line_position(i, j, number_of_horizontal_lines,
                       number_of_vertical_lines, shape_info, new_coordinates):
    """ We need to divide the image into parts. We'll call one part a box.
    The number of parts is set using
    the number_of_horizontal_lines and number_of_vertical_lines.
    The total number of boxes is number_of_horizontal_lines * number_of_vertical_lines.
    Boxes are numbered vertically (i) and horizontally (j).

    This function returns image coordinates of (i,j) box's center.

    Parameters
    ----------
    i: int
        The vertical number.

    j: int
        The horizontal number.

    number_of_horizontal_lines: int
        The number of horizontal part.

    number_of_vertical_lines: int
        The number of vertical part.

    shape_info: np.array
        Image shape, at the first place - height, the second - width.
            example: [224,400].

    new_coordinates: np.array, or list
         An array of x and y coordinates in a fixed coordinate system.

    Returns
    ----------
    np.array
       Coordinates of box center.
    """
    height, width = shape_info
    center_r_1 = [
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  int((i * (height) + height / 2) / number_of_horizontal_lines)]),
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  (int((i * (height) + height / 2) / number_of_horizontal_lines))]) +
        np.sign(new_coordinates[(int((i * (height) + height / 2) / number_of_horizontal_lines))]
                [int((j * (width) + width / 2) / number_of_vertical_lines)][:-1]) * [10, 0]]
    center_r_2 = [
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  int((i * (height) + height / 2) / number_of_horizontal_lines)]),
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  (int((i * (height) + height / 2) / number_of_horizontal_lines))]) +
        np.sign(new_coordinates[(int((i * (height) + height / 2) / number_of_horizontal_lines))]
                [int((j * (width) + width / 2) / number_of_vertical_lines)][:-1]) * [0, 10]]
    return center_r_1, center_r_2


def boxes_drawing(image, number_of_horizontal_lines,
                  number_of_vertical_lines, shape_info, i, j, new_coordinates):
    """ We need to divide the image into parts. We'll call one part box.
    The number of parts is set using
    the number_of_horizontal_lines and number_of_vertical_lines.
    The total number of boxes is number_of_horizontal_lines * number_of_vertical_lines.
    Boxes are numbered vertically (i) and horizontally (j).

    This function draws boxes on image.

    Parameters
    ----------
    image: np.array
         Image that is being drawn.

    i: int
        The vertical number.

    j: int
        The horizontal number.

    number_of_horizontal_lines: int
        The number of horizontal part.

    number_of_vertical_lines: int
        The number of vertical part.

    shape_info: np.array
        Image shape, at the first place - height, the second - width.
            example: [224,400].

    new_coordinates: np.array, or list
        An array of x and y coordinates in fixed coordinate system.

   Returns
   ----------
   np.array
      Image with drawn boxes.
   """
    height, width = shape_info
    cv2.rectangle(image,
                  (int(j * width / number_of_vertical_lines),
                   int(i * height / number_of_horizontal_lines)),
                  (int((j + 1) * width / number_of_vertical_lines),
                   int((i + 1) * height / number_of_horizontal_lines)),
                  (0, 0, 0), 1)
    center_r_1, center_r_2 = find_line_position(i, j,
                                                number_of_horizontal_lines,
                                                number_of_vertical_lines,
                                                image.shape[:-1], new_coordinates)
    cv2.arrowedLine(image, (int(center_r_1[0][0]), int(center_r_1[0][1]) - 5),
                    (int(center_r_1[1][0]), int(center_r_1[1][1] - 5)), (0, 0, 0), 1)
    cv2.arrowedLine(image, (int(center_r_2[0][0]), int(center_r_2[0][1]) - 5),
                    (int(center_r_2[1][0]), int(center_r_2[1][1] - 5)), (0, 0, 0), 1)
    return image


def define_center_coordinate(new_coordinates, i, j, shape_info, lines):
    """ We need to divide the image into parts. We'll call one part box.
    The number of parts is set using the number_of_horizontal_lines and number_of_vertical_lines.
    The total number of boxes is number_of_horizontal_lines * number_of_vertical_lines.
    Boxes are numbered vertically (i) and horizontally (j).

    This function returns coordinates of (i,j) box's center in fixed coordinate system.

    Parameters
    ----------
    i: int
        The vertical number.

    j: int
        The horizontal number.

    lines: dict
        Dict with number number_of_horizontal_lines and number_of_vertical_lines.
            example: {"horizontal": 5, "vertical": 4}.

    shape_info: np.array
        Image shape, at the first place - height, the second - width.
            example: [224,400].

    new_coordinates: np.array, or list
        An array of x and y coordinates in fixed coordinate system.

   Returns
   ----------
    np.array
       The coordinates of box center in a fixed coordinate system.
   """
    height, width = shape_info
    if are_infinity_coordinates(new_coordinates[int((i * (height) + height / 2) /
                                                    lines["horizontal"])]
                                [int((j * (width) + width / 2) / lines["vertical"])]):
        x_coordinate = "n/a"
        y_coordinate = "n/a"
    else:
        x_coordinate = round(
            new_coordinates[int((i * (height) + height / 2) / lines["horizontal"])]
            [int((j * (width) + width / 2) / lines["vertical"])][0])
        y_coordinate = round(
            new_coordinates[int((i * (height) + height / 2) / lines["horizontal"])]
            [int((j * (width) + width / 2) / lines["vertical"])][1])
    return x_coordinate, y_coordinate


def text_coordinates_on_frame(image, coordinates_value, i, j, shape_info, lines):
    """ We need to divide the image into parts. We'll call one part box.
    The number of parts is set using the number_of_horizontal_lines and number_of_vertical_lines.
    The total number of boxes is number_of_horizontal_lines * number_of_vertical_lines.
    Boxes are numbered vertically (i) and horizontally (j).

    This function writes fixed coordinates of (i,j) box's center on image.

    Parameters
    ----------
    image: np.array
        Image that is being drawn.

    i: int
        The vertical number.

    j: int
        The horizontal number.

    lines: dict
        Dict with number number_of_horizontal_lines and number_of_vertical_lines.
            example: {"horizontal": 5, "vertical": 4}.

    shape_info: np.array
        Image shape, at the first place - height, the second - width.
            example: [224,400].

    coordinates_value: np.array, or list.
        An array of x and y coordinates of boxes centers in a fixed coordinate system.

   Returns
   ----------
    np.array
        Image with drawn coordinates.
   """
    height, width = shape_info
    cv2.putText(image,
                'x {}'.format(coordinates_value[0]),
                (int((j * (width) + width / 2) / lines["vertical"] - 20),
                 int((i * (height) + height / 2) / lines["horizontal"] + 10)),
                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0),
                fontScale=1, thickness=1)
    cv2.putText(image,
                'y {}'.format(coordinates_value[1]),
                (int((j * (width) + width / 2) / lines["vertical"] - 10),
                 int((i * (height) + height / 2) / lines["horizontal"] - 7)),
                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0),
                fontScale=1, thickness=1)
    return image


def part_line(new_coordinates, image, number_of_horizontal_lines=4, number_of_vertical_lines=5):
    """ We need to divide the image into parts. We'll call one part box.
    The number of parts is set using the number_of_horizontal_lines and number_of_vertical_lines.
    The total number of boxes is number_of_horizontal_lines * number_of_vertical_lines.

    This function returns image with boxes and their centers coordinates in s fixed coordinate system.

    Parameters
    ----------
    image: np.array
        Image that is being drawn.

    number_of_horizontal_lines: int, optional
        The number of horizontal part.

    number_of_vertical_lines: int, optional
        The number of vertical part.

    new_coordinates: np.array, or list
        An array of x and y coordinates in fixed coordinate system.

   Returns
   ----------
    np.array
        Image with drawn boxes and their coordinates.
   """
    lines = {"horizontal": number_of_horizontal_lines, "vertical": number_of_vertical_lines, }
    for i in range(0, number_of_horizontal_lines):
        for j in range(0, number_of_vertical_lines):
            boxes_drawing(image, number_of_horizontal_lines,
                          number_of_vertical_lines, image.shape[:-1], i, j,
                          new_coordinates)
            coordinates_value = define_center_coordinate(new_coordinates,
                                                         i, j, image.shape[:-1], lines)
            image = text_coordinates_on_frame(image, coordinates_value,
                                              i, j, image.shape[:-1], lines)
    return image


def heatmap_frame_processing(new_coordinates, original_image, path_to_save, resize_width=400,
                             heatmap_constant=HEATMAP_CONSTANT):
    """ Here you can get heatmap visualization with visualization of a fixed coordinate system (FCS).
        For visualization of FCS, we need to divide the image into parts. We'll call one part box.
        The number of parts is set using the number_of_horizontal_lines and number_of_vertical_lines.
       The total number of boxes is number_of_horizontal_lines * number_of_vertical_lines.

       This function saves heatmap and fixed coordinate system visualizations.

    Parameters
    ----------
    new_coordinates: np.array, or list
        An array of x and y coordinates in fixed coordinate system.

    original_image: np.array
        Image that is being drawn.

    path_to_save: str
        Path to save getting image.

    resize_width: int, optional
        To speed up the performance of the script, pictures will be resized to this width.

    heatmap_constant: int, optional
        Normalization constant for heatmap.

   """
    image = imutils.resize(original_image, width=resize_width)
    heatmap = np.power(np.sum(np.power(new_coordinates, 2), axis=-1), 0.5)
    heatmap /= heatmap_constant
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + image
    part_line(new_coordinates, superimposed_img)
    cv2.imwrite(path_to_save, superimposed_img)


def make_template(shape):
    """ Return an array of each pixels coordinate.

    Parameters
    ----------
    shape: np.array, or list
        Template shape.

    Returns
    ----------
    np.array
        An array of each pixels coordinate.
   """

    pictures_template = np.array([[[0] * shape[2]] * shape[1]] * shape[0])
    for i in range(shape[0]):
        for j in range(shape[1]):
            pictures_template[i][j] = [j, i, 1]
    return pictures_template


def heatmap_video_processing(dict_with_homography_matrix, capture,
                             save_folder, resize_width=400, heatmap_constant=HEATMAP_CONSTANT):
    """ Here you can get heatmap visualization with
    visualization of a fixed coordinate system (FCS) for the whole video.
    The biggest displacement during the whole video is returned.

    Parameters
    ----------
    dict_with_homography_matrix:
        Dict with matrix superposition for each frame.
        If you want to analyze displacement only from the previous frame,
        you can specify homography dict without superposition.
            format: {frame_no: [],
                          ...
                      frame_no: []}

    capture: cv2.VideoCapture
        Analyzed video.

    save_folder: str
        The path to save pictures with heatmap visualization.

    resize_width: int
        To speed up the performance of the script, pictures will be resized to this width.

    heatmap_constant: int, optional
        Normalization constant for heatmap.

   Returns
   ----------
    float
        The biggest displacement during the whole video.
   """
    max_r = []
    _, original_image = capture.read()
    original_image = imutils.resize(original_image, resize_width)
    picture_template = make_template(original_image.shape)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for (frame_no, frame_H) in dict_with_homography_matrix.items():
        path_to_save = "{}/img{}.png".format(save_folder, "%06d" % frame_no)
        new_pictures = np.apply_along_axis(
            homography_transformation, 2, picture_template, frame_H)

        heatmap_frame_processing(new_pictures.tolist(), original_image,
                                 path_to_save, heatmap_constant=heatmap_constant)
        success, original_image = capture.read()
        if not success:
            continue
        original_image = imutils.resize(original_image, resize_width)
        max_r.append(np.max(new_pictures))
    return np.max(max_r)


def draw_original_and_fixed_coordinate(image, rect_original,
                                       rect_fixed, h_resize_coefficient, w_resize_coefficient):
    """ To draw original and fixed coordinate of the same point.

    Parameters
    ----------
    image: np.array
        Image that is being drawn.

    rect_original: dict
        The original coordinates of point.
            format: {"x1": float, "y1": float}.

    rect_fixed: dict
        The original coordinates of point.
            format: {"x1": float, "y1": float}.

    w_resize_coefficient: int
        Specify frame width on which homography matrix was received.

    h_resize_coefficient: int
        Specify frame height on which homography matrix was received.

   Returns
   ----------
    np.array
        Image with point and its coordinates in both coordinate system.
   """
    cv2.circle(image, (int(rect_original["x1"]), int(rect_original["y1"])),
               2, color=(255, 255, 255), thickness=5, lineType=8)
    cv2.putText(image,
                '{}{}'.format("original: ",
                              (int(rect_original["x1"] * h_resize_coefficient),
                               int(rect_original["y1"] * w_resize_coefficient))),
                (int(rect_original["x1"]), int(rect_original["y1"]) - 30),
                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                fontScale=1, thickness=1)
    cv2.putText(image,
                '{}{}'.format("fixed: ", (int(rect_fixed["x1"]), int(rect_fixed["y1"]))),
                (int(rect_original["x1"]), int(rect_original["y1"]) - 10),
                fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                fontScale=1, thickness=1)
    return image


def comparison_original_with_fixed_coordinate_video_processing(capture, original_coordinates,
                                                               recalculated_coordinates,
                                                               path_to_save, resize_info):
    """ To draw original and fixed coordinates of the same points on all frames from video.

    Parameters
    ----------
    capture: cv2.VideoCapture
        The analyzed video.

    original_coordinates: dict
        The original coordinates points.
            example: {frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}],
                     ...,
                     frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}]}.

    recalculated_coordinates: dict
        The fixed coordinates of points.
            example: {frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}],
                     ...,
                     frame_no: [{"x1": float, "y1": float}, ..., {"x1": float, "y1": float}]}.

    path_to_save: str
        Path to save received image.

    resize_info: dict
        An image shape which used for getting homography dictionary.
            format:{"h": (int), "w": (int))}.
   """
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    success, image = capture.read()
    frame_no = 1
    img_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_resize_coefficient = resize_info["h"] / img_h
    w_resize_coefficient = resize_info["w"] / img_w
    while success:
        if not success:
            continue
        for (rect_original, rect_fixed) in \
                zip(original_coordinates[frame_no], recalculated_coordinates[frame_no]):
            image = draw_original_and_fixed_coordinate(image, rect_original,
                                                       rect_fixed, h_resize_coefficient,
                                                       w_resize_coefficient)

        path_to_save_frame = "{}/img{}.png".format(path_to_save, "%06d" % frame_no)
        cv2.imwrite(path_to_save_frame, image)
        success, image = capture.read()
        frame_no += 1
