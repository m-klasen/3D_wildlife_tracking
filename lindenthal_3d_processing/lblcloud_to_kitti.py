import glob
import cv2
import os
import open3d as o3d
from pathlib import Path
import numpy as np
import math
import copy
import json
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt




def abs2rel_rotation(abs_rotation: float) -> float:
	"""Convert absolute rotation 0..360° into -pi..+pi from x-Axis.

	:param abs_rotation: Counterclockwise rotation from x-axis around z-axis
	:return: Relative rotation from x-axis around z-axis
	"""
	rel_rotation = np.deg2rad(abs_rotation)
	if rel_rotation > np.pi:
		rel_rotation = rel_rotation - 2 * np.pi
	return rel_rotation

def read_calib_file(filepath):
		''' Read in a calibration file and parse into a dictionary.
		Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
		'''
		data = {}
		with open(filepath, 'r') as f:
			for line in f.readlines():
				line = line.rstrip()
				if len(line)==0: continue
				key, value = line.split(':', 1)
				# The only non-float values in these files are dates, which
				# we don't care about anyway
				try:
					data[key] = np.array([float(x) for x in value.split()])
				except ValueError:
					pass

		return data

def project_to_image(pcd,pts_3d, calib):
	''' Project 3d points to image plane.
	
	'''
	R = np.array([[1, 0, 0],
				  [0, 0, 1],
				  [0,-1, 0]])
	# extrinsic transformation matrix
	T = np.array(calib["extrinsics"]).reshape(4,4)
	P = np.array(calib["P2"]).reshape(3,4)
	# convert to open3d pcl for easier handling
	box_pts = o3d.geometry.PointCloud()
	box_pts.points = o3d.utility.Vector3dVector(pts_3d)
	
	# first rotate Pcl towards camera plane, then transform with extrinsic
	if args.camera_model=="raspberry":
		box_pts.rotate(R.T, center=(0,0,0))
	elif args.camera_model=="realsense":
		box_pts.rotate(R.T, np.array(pcd.get_center()))
	box_pts.transform(T)

	# then project 3d point back using intrinics P to 2D image coordinates
	proj_pcl,_ = cv2.projectPoints(np.array(box_pts.points),
										np.array([0,0,0], np.float32),
										np.array([0,0,0], np.float32),P[:3,:3],None)
	proj_pcl = proj_pcl.squeeze(1)
	reorder = [3,0,2,5,6,1,7,4]
	proj_pcl = proj_pcl[reorder,:]
	return proj_pcl#[:,::-1]


def main(args):
	dataset_paths = sorted(os.listdir(args.root_dir))
 
	for ds_pth in dataset_paths:
		dataset_path = f'{args.root_dir}/{ds_pth}'

		pcl_labels_dir = str(Path(dataset_path) / "pcl_labels")
		pcl_labels = sorted(glob.glob(pcl_labels_dir+"/*"))
		if ds_pth == '20210818205614':
			pcl_dir = str(Path(dataset_path) / "pointcloud")
		else:
			pcl_dir = str(Path(dataset_path) / "pointcloud_dpt")
		pcls = sorted(glob.glob(pcl_dir+"/*"))
  
		cal_dir = str(Path(dataset_path) / "calib")
		calibs = sorted(glob.glob(cal_dir+"/*"))
  
		out_dir = f'{args.root_dir}/{ds_pth}'+"/label_2/"
		if not os.path.isdir(out_dir): os.mkdir(out_dir)

		for i,(pcl_fn,pcl_label,calib) in enumerate(zip(pcls,pcl_labels,calibs)):
			cali_matrices  = read_calib_file(calib)
			T = cali_matrices['extrinsics']
			T = np.array(T).reshape(4,4)
			# Read PCL, store backtfmed and original one
			pcl = o3d.io.read_point_cloud(pcl_fn)

			
			with open(pcl_label,"r") as f:
				ann_data = json.load(f)
			print(out_dir+pcl_label.split("/")[-1].replace(".json",".txt"))
			if len(ann_data['objects'])>0:
				out_file = open(out_dir+pcl_label.split("/")[-1].replace(".json",".txt"), 'w')	
				for obj in ann_data['objects']:
					obj_type = obj["name"]

					bbox = o3d.geometry.OrientedBoundingBox()

					bbox.center = [obj['centroid']['x'],
								obj['centroid']['y'],
								obj['centroid']['z']]
					bbox.extent = [obj['dimensions']['length'],
								obj['dimensions']['width'],
								obj['dimensions']['height']]
					r = obj['rotations']['z']	
					#print(bbox)
					if r != 0.0:
						box_angle = bbox.get_rotation_matrix_from_axis_angle([0,0,float(r)])
						bbox.R = box_angle
					pts_3d = np.asarray(bbox.get_box_points())
					pts_2d = project_to_image(pcl,pts_3d,cali_matrices)
					pts_2d = pts_2d[:,:2].astype(int)
					
				

					x1, y1  = np.amin(pts_2d, axis=0)
					x2, y2 = np.amax(pts_2d, axis=0)


									
					dimensions = " ".join([str(np.round(obj['dimensions']['length'],2)),
										str(np.round(obj['dimensions']['width'],2)),
										str(np.round(obj['dimensions']['height'],2))])
					location = " ".join([str(np.round(obj['centroid']['x'],2)),
										str(np.round(obj['centroid']['y'],2)),
										str(np.round(obj['centroid']['z'],2))])
					rotation_y = np.round(obj['rotations']['z'],3)
					a = np.round(float(rotation_y) - np.arctan2(-obj['centroid']['y'],obj['centroid']['x']),3)

					x1 = str(np.round(x1,2)); y1 = str(np.round(y1,2))
					x2 = str(np.round(x2,2)); y2 = str(np.round(y2,2))
	 
					str_to_srite = " ".join(
						[obj_type, "0 0",str(a),x1,y1,x2,y2, dimensions, location, str(rotation_y)])+ "\n"
					out_file.write(str_to_srite)
				


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--root_dir", type=str , default="/media/mlk/storage/data/train/pltsdorf/", 
  						help="path to directory containing video clips")
	parser.add_argument("--camera_model", type=str, default="realsense",)
	args = parser.parse_args()
	main(args)