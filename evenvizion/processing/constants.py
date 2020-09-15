"""
EvenVizion library.
https://github.com/AIHunters/EvenVizion

Supporting paper at:
https://github.com/AIHunters/EvenVizion/blob/master/EvenVizion-video_based_camera_localization_component.pdf
This is licensed under an MIT license. See the LICENSE.md file
for more information.

All the constants used in EvenVizion package.
"""

#: If x or y coordinate is more than this INFINITY_COORDINATE_FLAG,
#: the value of the coordinates is considered undefined.
INFINITY_COORDINATE = 10000

#: If the percentage of the number of points used for
#: calculating homography is less than this value, consider H is None.
LENGTH_ACCOUNTED_POINTS = 0.7

#: The threshold for OpenCV findHomography() function.
THRESHOLD_FOR_FIND_HOMOGRAPHY = 3.0

#: The ratio for Lowe's test.
LOWES_RATIO = 0.5

#: The minimum number of matching points you want to find.
MINIMUM_MATCHING_POINTS = 4

#: The heatmap normalization constant.

HEATMAP_CONSTANT = 1000
