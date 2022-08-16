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

def main(args):
    root = "/home/mlk/AMMOD/projects/OpenPCDet/data/ldth/training"
    root2 = "/media/mlk/storage/ldth2/training"
    
    
    for fold in os.listdir((f"{root}")):
        pcd_fns = os.listdir(f"{root}/{fold}/pointcloud_dpt/")
        img_fns = os.listdir(f"{root}/{fold}/color/")
        clb_fns = os.listdir(f"{root}/{fold}/calib/")
        lbl_fns = os.listdir(f"{root}/{fold}/label_2/")
        for pcd_fn,img_fn,clb_fn,lbl_fn in zip(pcd_fns,img_fns,clb_fns,lbl_fns):
            print(fold,pcd_fn)
            os.popen(f'cp {root}/{fold}/color/{img_fn} {root2}/image_2/{fold}_{img_fn}') 
            os.popen(f'cp {root}/{fold}/calib/{clb_fn} {root2}/calib/{fold}_{clb_fn}') 
            os.popen(f'cp {root}/{fold}/label_2/{lbl_fn} {root2}/label_2/{fold}_{lbl_fn}') 

            pcl = o3d.io.read_point_cloud(f"{root}/{fold}/pointcloud_dpt/{pcd_fn}")
            scan = np.concatenate((np.array(pcl.points), np.mean(np.array(pcl.colors),axis=-1)[:,None]),axis=-1)
            pcd_fn_out = f"{root2}/velodyne/{fold}_{pcd_fn.replace('.pcd','.bin')}"
            scan.astype('float32').tofile(pcd_fn_out)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)