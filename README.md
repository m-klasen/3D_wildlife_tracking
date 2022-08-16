# 3D Wildlife Tracking

This repository is an official implementation of the paper [3D Wildlife Tracking]().

## Installation 

To install required packages do 

```pip install -r requirements.txt ``` 

For visualization the custom Open3D-ML repository is required with 

`export OPEN3D_ML_ROOT=$ABS_PATH/Open3D-ML` to set as default path.

### Processing Data to pointclouds

​	0.1 Download the pretrained KITTI15 weights for  stereo matching from the [AANet](https://github.com/haofeixu/aanet) repository.

1. Go to `lindenthal_3d_processing/aanet` and run 

```
python predict_ldth.py \
--data_dir $path --output_dir $video_path \
--pretrained_aanet $model_path/aanet+_kitti15-2075aea1.pth \
--feature_type ganet --feature_pyramid --refinement_type hourglass --no_intermediate_supervision
```

​	to infer the stereo matching disparities

2. Go to `lindenthal_3d_processing/DPT`

3. Download the model weights from the DPT reporsitory.

4. ```
   python run_monodepth.py \
      -i $left_input frames folder  \
      -o $output -t dpt_hybrid_kitti
   ```

5. 

### OpenPCDet

1. install OpenPCDet via `python setup.py develop`

	2. To train the dataset run e.g. `cd tools` and `python train --cfg cfgs/ldthmodels/pointrcnn.yaml`

### Required Data Format

```
 ${ROOT_DIR}
  `-- |-- 202011221080502 				# videoclip
      `-- |-- color 					# list of color images
      	  |-- | -- xxx.jpg
      	  |-- depth_median_4			# list of depth maps
      	  |-- | -- xxx.exr
      	  |-- optical_flow				# generated optical flow
      	  |-- | -- xxx.flo
      	  |-- fg_out					# ground truth instances
      `-- instances_default.json 		# ground truth annotations
      `-- intrinsics.json				# Realsense Intrinsics
      `-- fg_predictions.json 			# Foreground Predictions 
      `-- detr_instance_predictions_conf_0.9.json
      									# 2D Instance Segmentation Predictions
```
**Format 3D MOT**

| Frame |    Type    |   2DBBOX (x1, y1, x2, y2)   | Score |   3D BBOX (h, w, l, x, y, z, rot_y)   | Alpha |
| ----- | :--------: | :-------------------------: | :---: | :-----------------------------------: | :---: |
| 0     | 1 (animal) | 726.4, 173.69, 917.5, 315.1 | 13.85 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, 0 |   0   |


| Frame | Track-ID |    Type    | Truncated | Occluded | Alpha |   2DBBOX (x1, y1, x2, y2)   |   3D BBOX (h, w, l, x, y, z, rot_y)   | Score |       RLE        |
| ----- | :------: | :--------: | :-------: | :------: | :---: | :-------------------------: | :-----------------------------------: | :---: | :--------------: |
| 0     |   0-N    | 1 (animal) |     0     |    0     |   0   | 726.4, 173.69, 917.5, 315.1 | 1.56, 1.58, 3.48, 2.57, 1.57, 9.72, 0 | [0-1] | 2D Mask Encoding |

### Running 3D Tracking

**Example of Inference**

```
python main.py --root_dir==$PATH_TO_CLIPS
```

**Example of Evaluation of 3D Mot Metrics**

```
python evaluation/evaluate_3dmot.py gt_aabbox_kalman 3D
```

**Example of Evaluation of 2D Mot Metrics**

```
python evaluation/evaluate_3dmot.py gt_aabbox_kalman 2D
```

Results will be printed and can be found under ```results/```
