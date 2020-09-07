import imutils

from EvenVizion.processing.frame_processing import *


def get_homography_dict(capture, resize_width=400, none_H_processing=True):
    """
    Main function for getting json with homography matrix for each frames
    """
    success, result_image = capture.read()
    characteristic_dict = {}
    if not success:
        raise ValueError("Problem with video! Can't read first frame")
    result_image = imutils.resize(result_image, width=resize_width)
    i = 1
    matrix_H_superposition = None
    matrix_H_prev = None
    matrix_H_first = True
    logging.info("Start video analysis")
    while success:
        i += 1
        frame_processing_a = FrameProcessing(result_image)
        success, image_b = capture.read()
        if not success:
            continue
        image_b = imutils.resize(image_b, width=resize_width)
        frame_processing_b = FrameProcessing(image_b)
        try:
            key_points_a, key_points_b = frame_processing_b.stitch(frame_processing_a)
            matrix_H = compute_homography(key_points_a, key_points_b, matrix_H_prev)
        except NoMatchesException:
            print(NoMatchesException)
            print("here")
            matrix_H = None
        except HomographyException:
            print(HomographyException)
            print("here_2")

            matrix_H = None

        result_image = image_b
        # matrix_H is None case processing
        if matrix_H is None:
            if none_H_processing:
                matrix_H = matrix_H_prev
            else:
                characteristic_dict[i] = {"H": None}
        ###
        characteristic_dict[i] = {"H": matrix_H.tolist()}
        matrix_H_superposition = \
            matrix_superposition(matrix_H, matrix_H_superposition, matrix_H_first)
        matrix_H_prev = matrix_H
        matrix_H_first = False
    characteristic_dict["resize_info"] = \
        {"h": result_image.shape[0], "w": result_image.shape[1]}
    return characteristic_dict
