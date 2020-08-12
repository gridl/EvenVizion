## About the project

EvenVizion - is a video-based camera localization component.

It allows evaluating the relative positions of objects and translating the coordinates of an object (relative to the frame) into a fixed coordinate system. To determine the position of an object, the main task was set – the creation of a fixed coordinate system. To solve this, we created the EvenVizion component. We show that the task can be solved even in bad filming conditions (sharp camera movement, bad weather conditions, filming in the dark and so on).

As a result you get JSON with the homography matrices for each frame.

More about the project structure, mathematical tools used and the explanation of visualization you can find in <a href="EvenVizion - video based camera localization component.pdf">EvenVizion - video based camera localization component.pdf</a>.

<img src='./experiment/test_video_processing/original_video_with_EvenVizion/original_video_with_EvenVizion.gif'>

## Installation

All the necessary libraries and versions you can find in requirements.txt.

## Running the code

### Get jsons with homography matrix

` $python3 evenvizion_component.py --path_to_video="test_video/test_video.mp4" --experiment_folder="experiment"  --experiment_name="test_video_processing" --path_to_original_coordinate="test_video/original_coordinates.json" `

All the parameters can be changed.

#### About the parameters:

- path_to_video - path to video that needs to be analyzed
- experiment_folder - folder to save all the results of the script running
- experiment_name - the name of an experiment 
- resize_width - to speed up the performance of the script, pictures will be resized to this width
- path_to_original_coordinate - if you want to get fixed object coordinates, specify the path to json with the original coordinate
- none_H_processing - there are some cases where Homography matrix can't be calculated, so you need to choose which script do you need to run in this case. If set True H = H on previous step, False - H = [1,0,0][0,1,0][0,0,1], it means there is no transformation on this frame
- number_of_matching_points- the minimum number of the matching points to find homography matrix
- show_matches - if you want to visualize matches, set True
- heatmap_visualization - for getting heatmap visualization, set True
- heatmap_constatnt - the constant for heatmap normalization
- number_of_vertical_lines - the number of vertical lines in heatmap visualization
- number_of_horizontal_lines - the number of horizontal lines in heatmap visualization


#### About the constants:

- INFINITY_COORDINATE_FLAG - if x or y coordinate is more than this threshold, the value of the coordinates is considered undefined
- LOWES_RATIO - the ratio for Lowe's test 
- THRESHOLD_FOR_FIND_HOMOGRAPHY -  the threshold for OpenCV findHomography() function
- LENGTH_ACCOUNTED_POINTS  - the constant for filter matching points



As a result, you get JSON with the matrix of a homography between two frames (not superposition), JSON with fixed coordinates and comparison between fixed and original coordinates.
- path to result jsons: experiment_folder + experiment_name
- path to fixed_coordinate_system_visualization: experiment_folder + experiment_name + fixed_coordinate_system_visualization

<img src='./experiment/test_video_processing/fixed_coordinate_system_visualization/fixed_coordinate_system_visualization.gif'>

- path to heatmap_visualization: experiment_folder + experiment_name + heatmap_visualization

### Visualize EvenVizion

`$python3 evenvizion_visualization.py --path_to_homography_dict="experiment/test_video_processing/dict_with_homography_matrix.json" --path_to_original_video="test_video/test_video.mp4" --experiment_name="visualize_camera_stabilization" --experiment_folder="experiment/test_video_processing"`

#### About the parameters:

- path_to_homography_dict - path to JSON with the homography dict

- path_to_original_video - path to video that needs to be analyzed

- experiment_name - experiment name

- experiment_folder - folder to save all the results

As a result, you get visualize_camera_stabilization_stabilization 
- path to  the result: experiment_folder + experiment_name

In this visualization, such changes as scaling and rotation are ignored. We took into consideration only the camera transition. But using the homography matrix from the previous step we can recalculate the coordinates considering all camera movements (Transition, scale, and rotation). You can see it in  heatmap_visualization.


### Compare EvenVizion with original video

`$python3 compare_evenvizion_with_original_video.py --path_to_original_video="test_video/test_video.mp4" --path_to_EvenVizion_result_frames="experiment/test_video_processing/visualize camera stabilization" --experiment_folder="experiment/test_video_processing" --experiment_name="original_video_with_EvenVizion"`

#### About the parameters:

- path_to_evenvizion_result_frames - path to EvenVizion visualization output frames

As a result you get original_video_with_EvenVizion visualization
- path to the result: experiment_folder + experiment_name

## KNOWN ISSUES

- N/A coordinates

The coordinates can’t be defined when the matching points are clinging to moving objects. This means that the filtration isn’t working well enough. The coordinates can’t be defined also when camera rotation angle is more than 90°. As a solution to the first problem we now consider applying video motion segmentation to distinguish static points from motion points (taking into consideration the nature of movement). As a solution to the second problem we see the transfer to the cylindrical coordinate system. 

- H=None

To find the homography you need at least 4 matching points. But in some cases the 4 points can’t be found, and the homography matrices are *Н=None*. In a current algorithm version we process such cases this way: if the argument *none_H_processing* is set for *True* we consider the matrix of the previous frame matches the matrix for the current frame. If set for *False*, then the matrix H is an identity matrix, meaning that there were no movement in the frame. It is necessary to think over a better handling of such situations.

- Error

There’s an inaccuracy in the coordinates. Poorly obtained homography matrix distorts the results of coordinate recalculation. The reasons for that are:
1. Poor filtration. If the points catch on a motion object, then the homography matrix will describe not only the camera movement, but also the independent movement of objects (for example, a person's walking).
2. Using the built-in *findHomography()* function of the OpenCV module. This function already assumes there is an error in the calculation of the homography matrix.

## Send us your failure cases and feedback!

Our project is open source and we will really appreciate getting your feedback!

We encourage the collaboration of any kind unless it violates our CODE_OF_CONDUCT and GitHub guidelines. 

If you find or resolve an issue, feel free to comment on GitHub or make a pull request and we will answer as soon as possible!

If you choose to use EvenVizion for your project, please do let us know by simply commenting here or emailing to rd@oxagile.com. 
