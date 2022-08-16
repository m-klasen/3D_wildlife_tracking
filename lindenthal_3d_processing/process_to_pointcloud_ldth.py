import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
import open3d as o3d
import cv2
import argparse

from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
from open3d._ml3d.utils import make_dir, DATASET
from skimage import segmentation
from skimage.feature import peak_local_max
import scipy.ndimage as ndi
import copy
import math
import matplotlib.pyplot as plt

from  open3d._ml3d.vis import BoundingBox3D


def crop_with_2dmask(o3dpc, mask, K=None):
    o3dpc = copy.deepcopy(o3dpc)
    cloud_npy = np.asarray(o3dpc.points)

    if K is None:
        mask = np.resize(mask, cloud_npy.shape[0])
        cloud_npy = cloud_npy[mask!=0]
        o3dpc = o3d.geometry.PointCloud()
        o3dpc.points = o3d.utility.Vector3dVector(cloud_npy)
    else:
        # project 3D points to 2D pixel
        cloud_npy = np.asarray(o3dpc.points)  
        x = cloud_npy[:, 0]
        y = cloud_npy[:, 1]
        z = cloud_npy[:, 2]
        px = np.uint16(x * K[0, 0]/z + K[0, 2])
        py = np.uint16(y * K[1, 1]/z + K[1, 2])
        # filter out the points out of the image
        H, W = mask.shape
        row_indices = np.logical_and(0 <= px, px < W-1)
        col_indices = np.logical_and(0 <= py, py < H-1)
        image_indices = np.logical_and(row_indices, col_indices)
        cloud_npy = cloud_npy[image_indices]
        mask_indices = mask[(py[image_indices], px[image_indices])]
        mask_indices = np.where(mask_indices != 0)[0]
        o3dpc.points = o3d.utility.Vector3dVector(cloud_npy[mask_indices])
    return o3dpc

def calc_ransac(pcl):
    # plane_model, inliers = pcl.segment_plane(distance_threshold=0.25,
    #                                          ransac_n=3000,
    #                                          num_iterations=1000)
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

def get_cam_params(args):
    if args.camera_model=="realsense":
        fx = 424.7448425292969 
        fy = 424.7448425292969
        ppx = 421.0166931152344
        ppy = 237.47096252441406
        h = 480
        w = 848
    elif args.camera_model=="plittersdorf":
        fx = 1591.4101 # 743.34
        fy = 1040.5182 # 743.34
        ppx = 960.0
        ppy = 384.0
        h = 994
        w = 1920
    return fx,fy,ppx,ppy,h,w

def get_roll(deg):
    return np.array([[1,           0.,          0.], 
                     [0., np.cos(deg),-np.sin(deg)], 
                     [0., np.sin(deg), np.cos(deg)]
                     ])
def get_pitch(deg):
    return np.array([[np.cos(deg), -np.sin(deg), 0.], 
                     [np.sin(deg),  np.cos(deg), 0.], 
                     [0.         ,           0,  1.]
                     ])


def main(args):

    dataset_paths = sorted(os.listdir(args.root_dir))
    for ds_pth in dataset_paths:
        dataset_path = f'{args.root_dir}/{ds_pth}'
        img_folder = "color"
        os.makedirs(f'{args.root_dir}/{ds_pth}/pointcloud_dpt', exist_ok=True)
        color_dir = f"{dataset_path}/{img_folder}"
        depth_dir = f"{dataset_path}/depth"

        color_files = [f for f in sorted(glob.glob(color_dir + "/*"))]
        depth_files = [f for f in sorted(glob.glob(depth_dir + "/*.exr"))]
        path_list   = color_files

        fx,fy,ppx,ppy,h,w = get_cam_params(args)
        cam = o3d.camera.PinholeCameraIntrinsic(w,h,fx,fy,ppx,ppy)
        pcl_tfm = [[1, 0, 0, 0], 
                   [0, 1, 0, 0], 
                   [0, 0, 1, 0], 
                   [0, 0, 0, 1]]
        extr = np.array([[1,          0.,          0.,   0.], 
                        [0., np.cos(0.3),-np.sin(0.3), -1.5], 
                        [0., np.sin(0.3), np.cos(0.3),  3.8], 
                        [0.,			0.,        0.,   1.]])
        deg = 0.15
        
        extr2 = np.array([[0., 0., 0.,  0.], 
                         [0., 0., 0., 1.], 
                         [0., 0., 0., 0], 
                         [0., 0., 0., 1.]])
        extr2[:3,:3] = get_roll(0.15) @ get_pitch(0.00)

        
        out_dir = dataset_path+"/calib/"
        if not os.path.isdir(out_dir): 
            os.mkdir(out_dir)
        else:
            calib_files = os.listdir(dataset_path+"/calib/")
            filepath = dataset_path+"/calib/" + calib_files[20]
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
            print("Loading extrinsics from file ...")
            extr2 = np.array(data['extrinsics']).reshape(4,4)
        b = 0.05
        intr = [fx,0,ppx,0,0,fy,ppy,0,0,0,1,0]
        extrinsic = list(extr2.reshape(-1))
        P_x = [0,0,0,0,0,0,0,0,0,0,0,0]
        R_0 = [1.,0.,0.,0.,1.,0.,0.,0.,1.]
        vel = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
		

        for idx in range(len(path_list)):
            color_file = color_files[idx]
            depth_file = depth_files[idx]
            
            
            fn = color_file.replace(img_folder,"calib")
            out_file = open(fn.replace(".jpg",".txt"), 'w')
            out_file.write("P0: ");out_file.write(" ".join(map(str, P_x))+ "\n")
            out_file.write("P1: ");out_file.write(" ".join(map(str, P_x))+ "\n")
            out_file.write("P2: ");out_file.write(" ".join(map(str, intr))+ "\n")
            out_file.write("P3: ");out_file.write(" ".join(map(str, P_x))+ "\n")
            out_file.write("R0_rect: ");out_file.write(" ".join(map(str, R_0))+ "\n")
            out_file.write("Tr_velo_to_cam: ");out_file.write(" ".join(map(str, vel))+ "\n")
            out_file.write("Tr_imu_to_velo: ");out_file.write(" ".join(map(str, vel))+ "\n")
            out_file.write("extrinsics: ");out_file.write(" ".join(map(str, extrinsic))+ "\n")
            out_file.close()
            
            depth = cv2.imread(depth_file, -1)
            color = np.array(cv2.imread(color_file)).astype(np.uint8)
            
            output_pcl_fn = f'{dataset_path}/pointcloud_dpt/{color_files[idx].split("/")[-1].replace(".jpg",".pcd")}'
            print(output_pcl_fn)
            depth = depth.astype(np.float32)
            if depth.shape == (280,720):
                d = np.zeros((480,848))
                d[200:,:720] = depth.copy()
                depth = d.astype(np.float32)
            else:
                d = np.zeros((480,848))
                d[200:,:720] = depth[200:,:720].copy()
                depth = d.astype(np.float32)
                depth[:200] = 0.; depth[:,720:] = 0.
            depth[depth>25.] = 0.

            color = o3d.geometry.Image(color) #o3d.io.read_image(color_file)
            depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1,depth_trunc=50)
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,cam,extr2)

            pcl.rotate([[1,0,0],
                        [0,0,1],
                        [0,-1,0]], center=(0,0,0))
            cl, ind = pcl.remove_statistical_outlier(nb_neighbors=40,std_ratio=2.0)
            pcl = pcl.select_by_index(ind) 

            o3d.io.write_point_cloud(output_pcl_fn, pcl, write_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str , default="/media/mlk/storage/AMMOD/ammod_realsense/data/2d/", help="path to directory containing video clips")
    parser.add_argument("--camera_model", type=str, default="realsense",)
    args = parser.parse_args()
    main(args)
