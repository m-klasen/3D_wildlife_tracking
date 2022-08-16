import glob
import cv2
import os
import open3d as o3d
from pathlib import Path
import numpy as np
import math
import pycocotools.mask as pycoco_mask
import copy
import json
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


def main(args, dataset_paths):
	out_file = open(f"/home/mlk/AMMOD/projects/OpenPCDet/data/ldth/ImageSets/{args.mode}.txt", 'w')

	for ds_pth in dataset_paths:
		dataset_path = f'{args.root_dir}/{ds_pth}'

		pcl_dir = str(Path(dataset_path) / "pointcloud_dpt")
		pcl = sorted(glob.glob(pcl_dir+"/*"))
		if len(pcl)==0:
			pcl_dir = str(Path(dataset_path) / "pointcloud")
			pcl = sorted(glob.glob(pcl_dir+"/*"))			

		
		for i,pcd_file in enumerate(pcl):
			fn = pcd_file.split("/")[-1].replace(".pcd","")
			
			out_file.write(f"{ds_pth}/{fn}"+ "\n")
	out_file.close()	
				
				


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--root_dir", type=str , default="/home/mlk/AMMOD/projects/OpenPCDet/data/ldth/training", help="path to directory containing video clips"
	)
	parser.add_argument("--mode", type=str, default="train")
	train_vids = ['20201217164720','20201219164321', '20201217170333', '20201218080334', '20201218165020', '20210818205614', '20201218080601']
	valid_vids = ['20210203175455', '20201217170908', '20201217172456', '20201220164721', '20201221080502']
	test_vids = os.listdir("/home/mlk/AMMOD/projects/OpenPCDet/data/ldth/testing")
	args = parser.parse_args()
	
	args.mode = "train"
	main(args, train_vids)
 
	args.mode = "val"
	main(args, valid_vids)

	args.mode = "test"
	main(args, test_vids)