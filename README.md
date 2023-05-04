# Loop closure detection algorithm

## Packages needed
>1. numpy----------1.24.2
>2. matplotlib-----3.6.2
>3. PyYAML---------6.0
>4. opencv-python--4.5.3
>5. onnxruntime----1.13.1

## How to run the code

1. Download [kitti dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. For testing, we use kitti 00 dataset, select image 300 to 600 for agent A and 3300 to 3600 for agent B.
When running the code, you need to change the path settings in the yaml file and create two folders to save images for agent A and B.

>1. 'general_path'is the path for original kitti dataset which saves all the images(...\kitti\dataset\sequences\00\image_0).
>2. 'A_mem_path' and 'B_mem_path' save the images acquired from agent A and B. In the simulation, I manually select 
300 images respectively from kitti 00 for A and B.
>3. 'pose_path' is the path for kitti pose information,(...\kitti\dataset\poses00.txt).
>4. 'storage_path' is the path where we save all the results like the loop closures and confusion matrix.

For 'A/B_start_label' and 'A/B_end_label', you change it according to the image label that you allocate to agent A and B.
For testing, we set it as 300/600, 3300/3600.

3. For the parameters in the yaml file.
>1. Threshold parameter: They are changeable for more accurate results. The detailed explanation is in the yaml file.
>2. Distance type: For now we use three types of formula to calculate the 'similarity' (cosine,l2 and exp). They are used 
in the searching process. You can switch to different formula in the coarse and fine search process. For now, we use 'l2'
in the coarse search and 'exp' in the fine search.

## Testing results
1. The code will save all the loop closures according to your 'storage_path'

   <img alt="image" height="150" src="https://github.com/wyp233/multi-agent-lcd/blob/main/results_1.png" width="200"/>
   <img alt="image" height="100" src="https://github.com/wyp233/multi-agent-lcd/blob/main/results_2.png" width="400"/>
2. It will draw the trajectory according to the pose information.

   <img alt="image" height="200" src="https://github.com/wyp233/multi-agent-lcd/blob/main/results_3.png" width="200"/>
3. The confusion matrix is calculated according to the ground truth table. We calculate the
ground truth table according to the pose info. If the distance between two location is smaller than 3 meter(predefined para),
we consider it as the same location. 
