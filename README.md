<img src='./EvenVizion/examples/test_video_processing/test_video/visualize_camera_stabilization/visualize_camera_stabilization.gif'>

------------------

EvenVizion - is a video-based camera localization Python package.

By using this package you can create a pipeline which allows evaluating the relative positions of objects and translating the coordinates of an object (relative to the frame) into a fixed coordinate system. To determine the position of an object, the main task was set – the creation of a fixed coordinate system. To solve this, we created the EvenVizion component. We show that the task can be solved even in bad filming conditions (sharp camera movement, bad weather conditions, filming in the dark and so on).

More about the package structure, mathematical tools used and the explanation of visualization you can find in <a href="EvenVizion-video_based_camera_localization_component.pdf">EvenVizion - video based camera localization component.pdf</a>.

## Installation
The most general installation is just to use pip, which should come with
any modern Python distribution.
 ```bash       
pip install evenvizion
```


#### Get the EvenVizion Source
If you prefer to download the source yourself

```bash
git clone https://github.com/AIHunters/EvenVizion.git
cd EvenVizion
```

#### Requirements
All the necessary libraries and versions you can find in requirements.txt.

## Basic use

[Here](https://github.com/AIHunters/EvenVizion/tree/master/EvenVizion/examples)  you can find examples of package usage.

### evenvizion_component.py
` $python3 evenvizion_component.py --path_to_video="test_video/test_video.mp4" --experiment_folder="experiment"  --experiment_name="test_video_processing" --path_to_original_coordinate="test_video/original_coordinates.json" `

All the parameters can be changed.

#### About the parameters:

- path_to_video - path to video that needs to be analyzed
- experiment_name - the name of an experiment 
- resize_width - to speed up the performance of the script, pictures will be resized to this width
- path_to_original_coordinate - if you want to get fixed object coordinates, specify the path to json with the original coordinate
- none_H_processing - there are some cases where Homography matrix can't be calculated, so you need to choose which script do you need to run in this case. If set True H = H on previous step, False - H = [1,0,0][0,1,0][0,0,1], it means there is no transformation on this frame
- heatmap_visualization - for getting heatmap visualization, set True
- show_matching_visualization - for getting matching visualization, set True

As a result, you get JSON with the matrix of a homography between two frames (not superposition), JSON with fixed coordinates and comparison between fixed and original coordinates.
- path to result jsons: experiment_folder + experiment_name
- path to fixed_coordinate_system_visualization: experiment_folder + video_name +  experiment_name + fixed_coordinate_system_visualization

<img src='./EvenVizion/examples/test_video_processing/test_video/fixed_coordinate_system_visualization/fixed_coordinate_system_visualization.gif'>

- path to heatmap_visualization: experiment_folder + experiment_name + video_name + heatmap_visualization

<img src='./EvenVizion/examples/test_video_processing/test_video/heatmap_visualization/heatmap.gif'>

- path to matching_visualization: experiment_folder + experiment_name + video_name +  heatmap_visualization

<img src='./EvenVizion/examples/test_video_processing/test_video/matching_visualization/matching_visualization.gif'>


### compare_evenvizion_with_original_video.py
`$python3 compare_evenvizion_with_original_video.py --path_to_homography_dict="experiment/test_video_processing/dict_with_homography_matrix.json" --path_to_original_video="test_video/test_video.mp4" --experiment_name="visualize_camera_stabilization" --experiment_folder="experiment/test_video_processing"`

#### About the parameters:

- path_to_homography_dict - path to JSON with the homography dict

- path_to_video - path to video that needs to be analyzed

- experiment_name - experiment name

As a result, you get visualize_camera_stabilization
- path to  the result: experiment_folder + experiment_name + video_name + visualize_camera_stabilization

<img src='./EvenVizion/examples/test_video_processing/test_video/visualize_camera_stabilization/visualize_camera_stabilization.gif'>


In this visualization, such changes as scaling and rotation are ignored. We took into consideration only the camera transition. But using the homography matrix from the previous step we can recalculate the coordinates considering all camera movements (Transition, scale, and rotation). You can see it in  heatmap_visualization.




## Licence
EvenVizion is licensed under the MIT license, as found in the LICENSE file.


## Send us your failure cases and feedback!

Our project is open source and we will really appreciate getting your feedback!

We encourage the collaboration of any kind unless it violates our CODE_OF_CONDUCT and GitHub guidelines. 

If you find or resolve an issue, feel free to comment on GitHub or make a pull request and we will answer as soon as possible!

If you choose to use EvenVizion for your project, please do let us know by simply commenting here or emailing to oss@aihunters.com. 


















