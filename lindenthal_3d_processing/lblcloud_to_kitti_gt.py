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
def project_to_image(pcd,pts_3d, calib, args):
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
	return proj_pcl

def get_cam_params(args):
	if args.camera_model=="realsense":
		fx = 424.7448425292969 
		fy = 424.7448425292969
		ppx = 421.0166931152344
		ppy = 237.47096252441406
		h = 480
		w = 848
	elif args.camera_model=="raspberry":
		fx = 1591.4101 # 743.34
		fy = 1040.5182 # 743.34
		ppx = 960.0
		ppy = 384.0
		h = 994
		w = 1920
	return fx,fy,ppx,ppy,h,w

def main(args):
	dataset_paths = sorted(os.listdir(args.root_dir))
 
	fx,fy,ppx,ppy,h,w = get_cam_params(args)
	R = np.array([[1,0,0],
				[0,0,1],
				[0,-1,0]])

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
		out_file = open(f"{ds_pth}.txt","w")

		for curr_frame_id,(pcl_fn,pcl_label,calib) in enumerate(zip(pcls,pcl_labels,calibs)):
			cali_matrices  = read_calib_file(calib)
			# Read PCL, store backtfmed and original one
			pcl = o3d.io.read_point_cloud(pcl_fn)

			
			with open(pcl_label,"r") as f:
				ann_data = json.load(f)
			print(out_dir+pcl_label.split("/")[-1].replace(".json",".txt"))
			if len(ann_data['objects'])>0:
					
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
					pts_2d = project_to_image(pcl,pts_3d,cali_matrices, args)
					pts_2d = pts_2d.astype(int)

					x1, y1  = np.amin(pts_2d, axis=0)
					x2, y2 = np.amax(pts_2d, axis=0)


									
					dimensions = " ".join([str(np.round(obj['dimensions']['length'],3)),
										str(np.round(obj['dimensions']['width'],3)),
										str(np.round(obj['dimensions']['height'],3))])
					location = " ".join([str(np.round(obj['centroid']['x'],3)),
										str(np.round(obj['centroid']['y'],3)),
										str(np.round(obj['centroid']['z'],3))])
					rotation_y = np.round(obj['rotations']['z'],3)
					track_id = obj['track_id']
					a = np.round(float(rotation_y) - np.arctan(obj['centroid']['x']/obj['centroid']['z']),3)

					x1 = str(np.round(x1,4)); y1 = str(np.round(y1,4))
					x2 = str(np.round(x2,4)); y2 = str(np.round(y2,4))
					str_to_srite = " ".join(
						[str(curr_frame_id), str(track_id), "deer", "0 0", str(a),x1,y1,x2,y2, dimensions, location, str(rotation_y)])+ "\n"
					out_file.write(str_to_srite)
		out_file.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--root_dir", type=str , default="/media/mlk/storage/data/valid/ldth/", 
  						help="path to directory containing video clips")
	parser.add_argument("--camera_model", type=str, default="raspberry",)
	args = parser.parse_args()
	main(args)
 
	#args = parser.parse_args(["--root_dir","/media/mlk/storage/AMMOD/ammod_realsense/data/2d/valid"])
	#main(args)
