import os

import imutils

from EvenVizion.processing.utils import *


def draw_matches(image_a, image_b, pts_a, pts_b):
    """
    Visualize matching points
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


def find_line_position(i, j, number_of_horizontal_lines, number_of_vertical_lines, shape_info, new_cordinates):
    height, width = shape_info
    center_r_1 = [
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  int((i * (height) + height / 2) / number_of_horizontal_lines)]),
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  (int((i * (height) + height / 2) / number_of_horizontal_lines))]) +
        np.sign(new_cordinates[(int((i * (height) + height / 2) / number_of_horizontal_lines))]
                [int((j * (width) + width / 2) / number_of_vertical_lines)][:-1]) * [10, 0]]
    center_r_2 = [
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  int((i * (height) + height / 2) / number_of_horizontal_lines)]),
        np.array([int((j * (width) + width / 2) / number_of_vertical_lines),
                  (int((i * (height) + height / 2) / number_of_horizontal_lines))]) +
        np.sign(new_cordinates[(int((i * (height) + height / 2) / number_of_horizontal_lines))]
                [int((j * (width) + width / 2) / number_of_vertical_lines)][:-1]) * [0, 10]]
    return center_r_1, center_r_2


def draw_one_box(image, number_of_horizontal_lines, number_of_vertical_lines, shape_info, i, j, new_coordinates):
    height, width = shape_info
    cv2.rectangle(image,
                  (int(j * width / number_of_vertical_lines), int(i * height / number_of_horizontal_lines)),
                  (int((j + 1) * width / number_of_vertical_lines),
                   int((i + 1) * height / number_of_horizontal_lines)),
                  (0, 0, 0), 1)
    center_r_1, center_r_2 = find_line_position(i, j, number_of_horizontal_lines, number_of_vertical_lines,
                                                image.shape[:-1], new_coordinates)
    cv2.arrowedLine(image, (int(center_r_1[0][0]), int(center_r_1[0][1]) - 5),
                    (int(center_r_1[1][0]), int(center_r_1[1][1] - 5)), (0, 0, 0), 1)
    cv2.arrowedLine(image, (int(center_r_2[0][0]), int(center_r_2[0][1]) - 5),
                    (int(center_r_2[1][0]), int(center_r_2[1][1] - 5)), (0, 0, 0), 1)
    return image


def define_center_coordinate(new_coordinates, i, j, shape_info, lines):
    height, width = shape_info
    if are_infinity_coordinates(new_coordinates[int((i * (height) + height / 2) /
                                                    lines["horizontal"])][
                                    int((j * (width) + width / 2) / lines["vertical"])]):
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


def draw_markup(new_coordinates, image, number_of_horizontal_lines=4, number_of_vertical_lines=5):
    """
    Function for drawing of boxes with centers coordinates on frame
    """
    lines = {"horizontal": number_of_horizontal_lines, "vertical": number_of_vertical_lines, }
    for i in range(0, number_of_horizontal_lines):
        for j in range(0, number_of_vertical_lines):
            draw_one_box(image, number_of_horizontal_lines, number_of_vertical_lines, image.shape[:-1], i, j,
                         new_coordinates)
            coordinates_value = define_center_coordinate(new_coordinates, i, j, image.shape[:-1], lines)
            image = text_coordinates_on_frame(image, coordinates_value, i, j, image.shape[:-1], lines)
    return image


def heatmap_frame_processing(new_coordinates, original_image, path_to_save, resize_width=400,
                             heatmap_constant=4000):
    """
    heatmap visualization
    """
    image = imutils.resize(original_image, width=resize_width)
    heatmap = np.power(np.sum(np.power(new_coordinates, 2), axis=-1), 0.5)
    heatmap /= heatmap_constant
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + image
    draw_markup(new_coordinates, superimposed_img)
    cv2.imwrite(path_to_save, superimposed_img)


def make_template(shape):
    """
    Make array with each pixels coordinate
    """
    pictures_template = np.array([[[0] * shape[2]] * shape[1]] * shape[0])
    for i in range(shape[0]):
        for j in range(shape[1]):
            pictures_template[i][j] = [j, i, 1]
    return pictures_template


def heatmap_video_processing(
        dict_with_homography_matrix, capture, save_folder, resize_width=400, heatmap_constant=1000):
    """
    Visualize heatmap with fixed coordinate system using all points in frame
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

        heatmap_frame_processing(new_pictures.tolist(), original_image, path_to_save, heatmap_constant=heatmap_constant)
        success, original_image = capture.read()
        if not success:
            continue
        original_image = imutils.resize(original_image, resize_width)
        max_r.append(np.max(new_pictures))
    return np.max(max_r)


def draw_point_coords(image, rect_original,
                      rect_fixed, h_resize_coefficient, w_resize_coefficient):
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

def compare_original_fixed_coordinate(capture, original_coordinates,
                                      recalculated_coordinates, path_to_save, resize_info):
    """
    Visualize two coordinate system on one frame
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
            image = draw_point_coords(image, rect_original,
                                      rect_fixed, h_resize_coefficient,
                                      w_resize_coefficient)

        path_to_save_frame = "{}/img{}.png".format(path_to_save, "%06d" % frame_no)
        cv2.imwrite(path_to_save_frame, image)
        success, image = capture.read()
        frame_no += 1
