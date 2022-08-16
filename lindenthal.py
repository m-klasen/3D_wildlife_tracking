import glob
import json
import logging
import math
import os
import pickle
import random
import sys
from os.path import abspath, dirname, exists, join
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycocotools.mask as pymasks
import scipy.ndimage as ndi
from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.utils import DATASET, make_dir
from open3d._ml3d.vis import BoundingBox3D
from plyfile import PlyData, PlyElement
from pycocotools import mask as pymask
from skimage import segmentation
from skimage.feature import peak_local_max
from sklearn.neighbors import KDTree
from tqdm import tqdm

from utils.mask_utils import *
from flow_tracking import *
from kalman_tracking import *

logging.basicConfig(
	level=logging.INFO,
	format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


def warp_flow(img, flow):
	h, w = flow.shape[:2]
	flow = -flow
	flow[:,:,0] += np.arange(w)
	flow[:,:,1] += np.arange(h)[:,np.newaxis]
	res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
	return res

def read_flow(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file', f)
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            data2D = data.reshape((h, w, 2))
    return data2D

DEBUG = False

class Lindenthal3DSplit():
	def __init__(self, dataset, split='all'):
		self.cfg = dataset.cfg
		color_files,depth_files,masks_files, calib_files = dataset.get_split_list(split)
		log.info("Found {} pointclouds for {}".format(len(color_files), split))
		self.path_list   = color_files
		self.color_files = color_files
		self.depth_files = depth_files
		self.masks_files = masks_files
		self.calib_files = calib_files
		self.box_centers = []
		self.trace_data = np.array([[[0],[0],[0]]])
		self.dataset = dataset
		self.split = split
		self.axis_aligned = self.dataset.axis_aligned
		self.axis_alignment = 'aabbox' if self.axis_aligned else 'oobbox'
		self.mode = self.cfg.mode
		if self.cfg.segm_mode=='inst_segm':
			with open(self.cfg.inst_segm_path, "r") as f: self.pred_inst_seg = json.load(f)
		elif self.cfg.segm_mode=='watershed' or self.cfg.segm_mode=='dbscan':
			with open(self.cfg.fg_path, "r") as f: self.fg_preds = json.load(f)

		fx = 424.7448425292969 
		fy = 424.7448425292969
		h = 480
		ppx = 421.0166931152344
		ppy = 237.47096252441406
		w = 848
		self.cam = o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,ppx,ppy)
		self.pcl_tfm = [[1, 0, 0, 0], 
						[0,-1, 0, 0], 
						[0, 0,-1, 0], 
						[0, 0, 0, 1]]
		self.extr = np.array([[1, 0., 0.,  0.], 
					[0., np.cos(0.3),  -np.sin(0.3), 0.], 
					[0., np.sin(0.3), np.cos(0.3), 0.], 
					[0.,            0., 0., 1.]])
		# self.extr = np.array([[1, 0., 0., 0.], 
		# 					  [0., 1,  0., 0.], 
		# 					  [0., 0., 1,  0.], 
		# 					  [0., 0., 0., 1.]])
		#Orient based on last frame ransac plane fitting
		color_file = self.color_files[-1]
		depth_file = self.depth_files[-1]

		depth = np.zeros((480,848),dtype=np.uint16)
		try:
			depth[200:,:720] = (cv2.imread(depth_file, -1)*1000).astype(np.uint16)
		except:
			depth = (cv2.imread(depth_file, -1)*1000).astype(np.uint16)
		color = o3d.io.read_image(color_file)
		depth = o3d.geometry.Image(depth)
		rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1000,depth_trunc=25)
		pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,self.cam,self.extr)
		pcl.transform(self.pcl_tfm)
		self.R = self.calc_ransac(pcl)
		self.box_centers, self.data_dict, self.trace_data = self.process_video() 

	def __len__(self):
		return len(self.path_list)


	def get_data(self, idx):
		color_file = self.color_files[idx]
		if not os.path.exists(f'{self.dataset.dataset_path}{self.cfg.pcl_path}'):
			depth_file = self.depth_files[idx]
			color = o3d.io.read_image(color_file)
			depth = o3d.geometry.Image((cv2.imread(depth_file, -1)*1000).astype(np.uint16))
			rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1000,depth_trunc=100)
			pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,self.cam,self.extr)
			pcl.transform(self.pcl_tfm)
			#pcl.rotate(self.R, center=(0,0,0))
		else:
			pcd_path = self.color_files[idx].replace("color",self.cfg.pcl_path).replace(".jpg",".pcd")
			pcl = o3d.io.read_point_cloud(pcd_path)

		fx, fy = self.cam.get_focal_length()
		ppx,ppy = self.cam.get_principal_point()	

		file_idx = os.path.relpath(self.color_files[idx], self.cfg.root_dir)
		

		#GET 2D MASK
		confidences = [];boxes_2d=[]
		if self.cfg.segm_mode=="inst_segm":
			inst_masks, boxes_2d, confidences = get_instance_segm_masks(self.cfg,self.pred_inst_seg,file_idx)
		elif self.cfg.segm_mode=="watershed":
			inst_masks, boxes_2d = watershed_fg_preds_instances(self.cfg,self.fg_preds,file_idx)
		elif self.cfg.segm_mode=="dbscan":
			inst_masks, boxes_2d = dbscan_fg_preds_instances(self.cfg,pcl,self.fg_preds,file_idx)

		else:
			masks_file = self.masks_files[idx]
			instance_masks = None
			inst_masks = cv2.imread(masks_file,0)[200:,:720]
			for mask_idx in np.unique(inst_masks)[1:]:
				instance_masks = np.concatenate((instance_masks,inst_masks[None,:,:]==mask_idx),axis=0)\
										if instance_masks is not None else inst_masks[None,:,:]==mask_idx
				box = pymasks.toBbox(pymask.encode(np.asfortranarray(inst_masks==mask_idx)))
				if box[0]<720:
					boxes_2d.append(box)
			inst_masks = instance_masks

		bboxes, theta = self.get_bboxes(inst_masks,pcl)

		cl, ind = pcl.remove_statistical_outlier(nb_neighbors=10,std_ratio=2.0)
		pcl = pcl.select_by_index(ind)
		points = np.array(pcl.points, dtype=np.float32)
		color = np.array(pcl.colors, dtype=np.float32)

		if isinstance(inst_masks,np.ndarray):
			assert inst_masks.shape[1:] == (280,720)
			assert inst_masks.shape[0]  == len(boxes_2d)
		data = {'name': str(color_file.split('/')[0]),
				'point': points, 
				'feat': color,
				'bounding_boxes': bboxes,
				'theta': theta,
				'tracks': np.array(self.trace_data[:idx+1,:,:idx+1]),
				'inst_masks': inst_masks,
				'boxes_2d': boxes_2d,
				'inst_masks_conf': confidences if confidences else None,
				}

		return data

	def process_video(self):
		nmb_tracks = 300
		full_clip = self.path_list

		KalmanBoxTracker.count = 0
		ParticleBoxTracker.count = 0
		tracker = Tracker(iou=self.cfg.iou3d)


		if self.mode=="2D":
			if self.cfg.save_eval:
				save_root = "results"; result_sha = self.dataset.dataset_path.split("/")[-2]
				save_dir = os.path.join(save_root, f'{self.cfg.mode}/{self.cfg.segm_mode}_{self.cfg.depth_path}_{self.axis_alignment}_{self.cfg.tracking_mode}_iou{self.cfg.iou3d}'); 
				os.makedirs(save_dir, exist_ok=True)
				eval_dir = os.path.join(save_dir, "data"); 
				os.makedirs(eval_dir, exist_ok=True)
				save_trk_dir = os.path.join(save_root, save_dir, 'trk_withid', result_sha); 
				os.makedirs(save_trk_dir, exist_ok=True)
				eval_file = os.path.join(eval_dir,f'{result_sha}.txt'); print(eval_file); 
				eval_file = open(eval_file, 'w')
			boxes_center = np.full((nmb_tracks,len(full_clip),3),-1,dtype=np.float32)
			traces_data = np.full((len(full_clip),nmb_tracks,len(full_clip),3),-1,dtype=np.float32)
			
			data_dict_ = {}
			data_dict_['bbox_center']		= np.full((len(full_clip),nmb_tracks,3),-1,dtype=np.float32)
			data_dict_['bbox_size']			= np.full((len(full_clip),nmb_tracks,3),-1,dtype=np.float32)
			data_dict_['bbox_theta']		= np.full((len(full_clip),nmb_tracks,1),-1,dtype=np.float32)
			data_dict_['2d_bbox']			= np.full((len(full_clip),nmb_tracks,4),-1,dtype=np.float32)
			data_dict_['2d_bbox_conf']		= np.full((len(full_clip),nmb_tracks,1),-1,dtype=np.float32)
		elif self.mode=="3D":
			self.detections_3d_path = [f'pointrcnn_predictions_v0.3/{p.split("/")[-3]}/label_2/{p.split("/")[-1].replace(".jpg",".txt")}' for p in self.path_list]
			
			if self.cfg.save_eval:
				save_root = "results"; result_sha = self.dataset.dataset_path.split("/")[-2]
				#save_dir = os.path.join(save_root, f'{self.cfg.mode}/_iou{self.cfg.iou3d}'); 
				save_dir = os.path.join(save_root, f'3D/iou{self.cfg.iou3d}'); 
				os.makedirs(save_dir, exist_ok=True)
				eval_dir = os.path.join(save_dir, "data"); 
				os.makedirs(eval_dir, exist_ok=True)
				save_trk_dir = os.path.join(save_dir, 'trk_withid', result_sha); 
				os.makedirs(save_trk_dir, exist_ok=True)
				eval_file = os.path.join(eval_dir,f'{result_sha}.txt'); print(eval_file); eval_file = open(eval_file, 'w')

		for i in range(len(full_clip)):
			if self.cfg.save_eval:
				save_trk_file = os.path.join(save_trk_dir, self.path_list[i].split("/")[-1].replace(".jpg",".txt")); 
				save_trk_file = open(save_trk_file, 'w')
			if self.mode=="2D":
				data = self.get_data(i)
				inst_masks = data['inst_masks']
				boxes_2d   = np.array(data['boxes_2d'])

				mask_ids = np.unique(inst_masks)

				bboxes 		= data['bounding_boxes']
				thetas 		= data['theta']

				#for m in inst_masks:
				#	plt.imshow(m);plt.show()

				"""

				"""
				num_bboxes = len(bboxes)
				if num_bboxes>0:
					data_dict_['bbox_center'][i,:num_bboxes] = [bbox.center for bbox in bboxes]
					data_dict_['bbox_size'][i,:num_bboxes] = [bbox.size for bbox in bboxes]
					data_dict_['bbox_theta'][i,:num_bboxes,:] = np.array(thetas)[:,None]


					boxes_2d[:,2]+= boxes_2d[:,0];boxes_2d[:,3]+= boxes_2d[:,1]
					data_dict_['2d_bbox'][i,:num_bboxes] = boxes_2d
					data_dict_['2d_bbox_conf'][i,:num_bboxes] = np.array(data['inst_masks_conf'])[:,None]



				### Tracking ###
				# Get all detected box centers
				det_idx = np.where(data_dict_['bbox_center'][i,:,0]!=-1)[0]\
						& np.where(data_dict_['bbox_center'][i,:,1]!=-1)[0]\
						& np.where(data_dict_['bbox_center'][i,:,2]!=-1)[0]

				x = np.concatenate((data_dict_['bbox_center'],data_dict_['bbox_size'],data_dict_['bbox_theta']), axis=2)
				if (len(det_idx) > 0):
					# Track object using Kalman Filter
					dets_all = {'dets': x[i,det_idx], 
								'info': np.concatenate((data_dict_['2d_bbox'][i,det_idx],data_dict_['2d_bbox_conf'][i,det_idx]), axis=1)}
					if self.cfg.tracking_mode == 'kalman':
						trackers = tracker.update(dets_all)
						if len(det_idx) > 0: 
							print("FrameID", i, "Detections: ", det_idx , "Track ids: ", [d[7] for d in trackers])
					for d in trackers:
						bbox3d_tmp = d[0:7]	   # x y z w h l theta, theta in camera coordinate
						id_tmp = d[7]
						ori_tmp = 0.
						type_tmp = 'goat'
						bbox2d_tmp_trk = d[8:12]
						conf_tmp = d[12]
						track_id = np.array(d[7], dtype=np.uint8)
						curr_trace = traces_data[i-1,track_id]

						if curr_trace.size > 0: 
							traces_data[i,track_id,:curr_trace.shape[0]] = curr_trace
							traces_data[i,track_id,i]  = d[0:3]

						if self.cfg.save_eval:
							# save in detection format with track ID, can be used for dection evaluation and tracking visualization
							str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
							bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
							bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
							save_trk_file.write(str_to_srite)

							# save in tracking format, for 3D MOT evaluation
							str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (i, id_tmp, 
								type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
								bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2],  bbox3d_tmp[6], 
								conf_tmp)
							eval_file.write(str_to_srite)
				
			elif self.mode=="3D":
				fp = self.detections_3d_path[i]
				if os.path.isfile(fp):
					seq_dets = []
					classes = []
					with open(fp, 'r') as f:
						for line in f.readlines():
							line = line.rstrip().split(" ")
							classes.append(line.pop(0))
							seq_dets.append(line)
					seq_dets = np.array(seq_dets, dtype=np.float32)
					if seq_dets.size > 0:
						conf = seq_dets[:, -1].reshape((-1, 1))		# confidence
						c    = classes	# class
						ori_array = seq_dets[:, -2].reshape((-1, 1))		# orientation
						bbox2d = seq_dets[:, 3:7] 		# other information, e.g, 2D box, ...
						#additional_info = np.concatenate((ori_array, other_array), axis=1)		
						dets = seq_dets[:, 7:14]            # h, w, l, x, y, z, theta in camera coordinate follwing KITTI convention
						reorder = [3,4,5,0,1,2,6]
						dets = dets[:,reorder]
						dets_all = {'dets': dets, 'info': np.concatenate((ori_array,bbox2d,conf), axis=-1)}


						trackers = tracker.update(dets_all)
						if seq_dets.shape[0] > 0: 
							print("FrameID", i, "Detections: ", seq_dets.shape[0] , "Track ids: ", [d[7] for d in trackers])

						if DEBUG:
							data = self.get_data(i)
							pcl = o3d.geometry.PointCloud()
							pcl.points = o3d.utility.Vector3dVector(data['point'])
							preds = []
							for det in dets:
								x, y, z, w, h, l, theta = det[0:7]
								bbox = o3d.geometry.OrientedBoundingBox()  
								bbox.center = [x,y,z]
								bbox.extent = [w,h,l]
								bbox.color = [1,0,0]
								if theta != 0.0:
									box_angle = bbox.get_rotation_matrix_from_axis_angle([0,0,float(theta)])
									bbox.R = box_angle
								preds.append(bbox)
							boxes = []
							for d in trackers:
								x, y, z, w, h, l, theta = d[0:7]
								bbox = o3d.geometry.OrientedBoundingBox()  
								bbox.center = [x,y,z]
								bbox.extent = [w,h,l]
								bbox.color = [0,0,1]
								if theta != 0.0:
									box_angle = bbox.get_rotation_matrix_from_axis_angle([0,0,float(theta)])
									bbox.R = box_angle
								boxes.append(bbox)

							mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0, 0, 0])
							o3d.visualization.draw_geometries([pcl,*preds, *boxes, mesh_frame],)

						for d in trackers:
							bbox3d_tmp = d[0:7]	   # x y z w h l theta, theta in camera coordinate
							id_tmp = d[7]
							ori_tmp = d[8]
							type_tmp = 'goat'
							bbox2d_tmp_trk = d[9:13]
							conf_tmp = d[13]
							track_id = np.array(d[7], dtype=np.uint8)

							if self.cfg.save_eval:
								# save in detection format with track ID, can be used for dection evaluation and tracking visualization
								str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
								bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
								bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[6], conf_tmp, id_tmp)
								save_trk_file.write(str_to_srite)

								# save in tracking format, for 3D MOT evaluation
								str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (i, id_tmp, 
									type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
									bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2],  bbox3d_tmp[6], 
									conf_tmp)
								eval_file.write(str_to_srite)
				boxes_center,data_dict_, traces_data = None,None,None

		return boxes_center,data_dict_, traces_data

	def calc_ransac(self,pcl):
		# plane_model, inliers = pcl.segment_plane(distance_threshold=0.25,
		#										  ransac_n=3000,
		#										  num_iterations=1000)
		plane_model = [-0.00,0.72,0.40,6.96]
		[a, b, c, d] = plane_model

		# Translate plane to coordinate center
		pcl.translate((0,-d/c,0))

		# Calculate rotation angle between plane normal & z-axis
		plane_normal = tuple(plane_model[:3])
		z_axis = (0,0,1)
		def vector_angle(u, v):
			return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))
		rotation_angle = vector_angle(plane_normal, z_axis)

		# Calculate rotation axis
		plane_normal_length = math.sqrt(a**2 + b**2 + c**2)
		u1 = b / plane_normal_length
		u2 = -a / plane_normal_length
		rotation_axis = (u1, u2, 0)

		# Generate axis-angle representation
		optimization_factor = 1.4
		axis_angle = tuple([x * rotation_angle * optimization_factor for x in rotation_axis])

		# Rotate point cloud
		R = pcl.get_rotation_matrix_from_axis_angle(axis_angle)
		return R

	def get_flow_centers(self,inst_masks,pcl):
		#################################################################################
		inst_pcd_bbox=[]

		inst_flow_list = [crop_with_2dmask(pcl,mask) for mask in inst_masks]
		
		for flow_inst in inst_flow_list:
			cl, ind = flow_inst.remove_statistical_outlier(nb_neighbors=30,
															std_ratio=1.0)
			pcl_processed = flow_inst.select_by_index(ind)
			#if np.array(pcl_inst.points).shape[0] > 20:
			if self.axis_aligned:
				inst_pcd_bbox.append(pcl_processed.get_axis_aligned_bounding_box())
			else:
				inst_pcd_bbox.append(pcl_processed.get_oriented_bounding_box())
		lines_boxes_3d = []
		theta = []
		for box in inst_pcd_bbox:
			if not self.axis_aligned:
				og_r = box.R.copy()
				yaw = -np.arctan2(og_r[0,2],[og_r[1,2]])
				r = np.linalg.inv(box.R)
				new_r = [[np.cos(yaw),-np.sin(yaw), 0],
						[np.sin(yaw), np.cos(yaw), 0],
						[		  0,		   0, 1]]
				box.rotate(r)
				box.rotate(new_r)

				yaw = -yaw
				left = [np.cos(yaw), -np.sin(yaw), 0]
				# y-axis
				front = [np.sin(yaw), np.cos(yaw), 0]
				# z-axis
				up = [0, 0, 1]
				size = np.array(box.extent)
			else:
				# x-axis
				left = [1, 0, 0]
				# y-axis
				front = [0, 1, 0]
				# z-axis
				up = [0, 0, 1]
				size = np.array(box.get_extent())
				yaw=0.
			center = np.array(box.get_center())
			
			size[1], size[2] = size[2], size[1]
			lines_boxes_3d.append(BoundingBox3D(center,front,up,left,size,'Unclassified',-1))	
			theta.append(yaw)
		return lines_boxes_3d, theta

	def get_bboxes(self,inst_masks,pcl):
		inst_pcd_list = [crop_with_2dmask(pcl,mask) for mask in inst_masks]
		inst_pcd_bbox = []
		for i,pcl in enumerate(inst_pcd_list):
			cl, ind = pcl.remove_statistical_outlier(nb_neighbors=30,std_ratio=1.0)
			pcl_processed = pcl.select_by_index(ind)
			#if np.array(pcl_processed.points).shape[0] > 20:
			if self.axis_aligned:
				inst_pcd_bbox.append(pcl_processed.get_axis_aligned_bounding_box())
			else:
				inst_pcd_bbox.append(pcl_processed.get_oriented_bounding_box())
		
		lines_boxes_3d = []
		theta = []
		for box in inst_pcd_bbox:
			if not self.axis_aligned:
				og_r = box.R.copy()
				yaw = -np.arctan2(og_r[0,2],[og_r[1,2]])
				r = np.linalg.inv(box.R)
				new_r = [[np.cos(yaw),-np.sin(yaw), 0],
						[np.sin(yaw), np.cos(yaw), 0],
						[		  0,		   0, 1]]
				box.rotate(r)
				box.rotate(new_r)

				yaw = -yaw
				left = [np.cos(yaw), -np.sin(yaw), 0]
				# y-axis
				front = [np.sin(yaw), np.cos(yaw), 0]
				# z-axis
				up = [0, 0, 1]
				size = np.array(box.extent)
			else:
				# x-axis
				left = [1, 0, 0]
				# y-axis
				front = [0, 1, 0]
				# z-axis
				up = [0, 0, 1]
				size = np.array(box.get_extent())
				yaw=0.
			center = np.array(box.get_center())
			
			size[1], size[2] = size[2], size[1]
			lines_boxes_3d.append(BoundingBox3D(center,front,up,left,size,'Unclassified',-1))	
			theta.append(yaw)
		return lines_boxes_3d, theta

	def get_attr(self, idx):
		pc_path = self.path_list[idx]
		name = Path(pc_path).name.split('.')[0]

		attr = {'name': name, 'path': pc_path, 'split': self.split}
		return attr


class Lindenthal3D(BaseDataset):
	"""   
	Args:
		dataset_path: The path to the dataset to use.
		name: The name of the dataset.
		cache_dir: The directory where the cache is stored.
		use_cache: Indicates if the dataset should be cached.
		num_points: The maximum number of points to use when splitting the dataset.
		ignored_label_inds: A list of labels that should be ignored in the dataset.
		test_result_folder: The folder where the test results should be stored.
	"""

	def __init__(self,
				 root_dir,
				 dataset_path,
				 name='Lindenthal',
				 mode="2D",
				 axis_aligned=True,
				 tracking_mode='scene_flow',
				 segm_mode='gt',
				 gt_path="fg_out",
				 images_path="color",
				 depth_path="depth_median_4",
				 calib_path="calib",
				 pcl_path="point_cloud",
				 fg_path=None,
				 inst_segm_path=None,
				 iou3d=iou3d,
				 save_eval=True,
				 cache_dir='logs/cache',
				 use_cache=False,
				 num_points=201600,
				 ignored_label_inds=[],
				 test_result_folder='./test',
				 **kwargs):

		super().__init__(dataset_path=os.path.join(root_dir,dataset_path),
						 name=name,
						 cache_dir=cache_dir,
						 use_cache=use_cache,
						 num_points=num_points,
						 ignored_label_inds=ignored_label_inds,
						 test_result_folder=test_result_folder,
						 **kwargs)

		cfg = self.cfg
  
		self.cfg.mode = mode

		self.dataset_path = cfg.dataset_path
		self.cfg.root_dir = root_dir
		self.cfg.save_eval = save_eval
		self.cfg.iou3d = iou3d
		self.label_to_names = self.get_label_to_names()

		self.axis_aligned = axis_aligned
		self.cfg.tracking_mode = tracking_mode
		self.cfg.segm_mode = segm_mode
		self.cfg.inst_segm_path = os.path.join(root_dir,inst_segm_path)
		self.cfg.calib_path = calib_path
		self.cfg.fg_path = os.path.join(root_dir,fg_path)
		self.cfg.gt_path = gt_path
		self.cfg.images_path = images_path
		self.cfg.depth_path = depth_path
		self.cfg.pcl_path = pcl_path

		self.color_dir = str(Path(cfg.dataset_path) / self.cfg.images_path)
		self.depth_dir = str(Path(cfg.dataset_path) / self.cfg.depth_path)
		if self.cfg.segm_mode=='gt' or self.cfg.segm_mode=='dbscan' or self.cfg.segm_mode=='watershed':
			self.masks_dir = str(Path(cfg.dataset_path) / self.cfg.gt_path)
			self.masks_files = [f for f in sorted(glob.glob(self.masks_dir + "/*"))]
		else: self.masks_files = None
		self.color_files = [f for f in sorted(glob.glob(self.color_dir + "/*"))]
		self.depth_files = [f for f in sorted(glob.glob(self.depth_dir + "/*"))]
		self.calib_files = [f for f in sorted(glob.glob(str(Path(cfg.dataset_path) / self.cfg.calib_path) + "/*"))]
		

	@staticmethod
	def get_label_to_names():
		label_to_names = {
			0: 'Unclassified',
		}
		return label_to_names

	def get_split(self, split):
		return Lindenthal3DSplit(self, split=split)

	def get_split_list(self, split):
		return self.color_files, self.depth_files,self.masks_files, self.calib_files

	def is_tested(self, attr):
		return False

	def save_test_result(self, results, attr):
		return False



DATASET._register_module(Lindenthal3D)
