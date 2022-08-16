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

from datumaro.components.extractor import Cuboid3d
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem


def crop_with_2dmask(o3dpc, mask, K=None):
    """ crop open3d point cloud with given 2d binary mask
    Args: 
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
        mask (np.array): binary mask aligned with the point cloud frame shape of [H, W]
        K (np.array): intrinsic matrix of camera shape of (4x4)
        if K is not given, point cloud should be ordered
    Returns:
        o3dpc (open3d.geometry.PointCloud): filtered open3d point cloud
    """
    o3dpc = copy.deepcopy(o3dpc)
    cloud_npy = np.asarray(o3dpc.points)

    if K is None:
        mask = np.resize(mask, cloud_npy.shape[0])
        cloud_npy = cloud_npy[mask !=0]
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


def main(args):
    dataset_paths = sorted(os.listdir(args.root_dir))
    for ds_pth in dataset_paths:
        dataset_path = f'{args.root_dir}/{ds_pth}'
        pcd_dir = str(Path(dataset_path) / "point_cloud")
        img_dir = str(Path(dataset_path) / "color")
        
        with open(f"{dataset_path}/instances_default.json", "r") as f: annos = json.load(f)
        pcd_files = [f for f in sorted(glob.glob(pcd_dir+"/*"))]
        img_files = [f for f in sorted(glob.glob(img_dir+"/*"))]
        ann_id = 0
  
        dataset_items = []
        for i,(pcd_file, img_fn) in enumerate(zip(pcd_files,img_files)):
            fn = pcd_file.split("/")[-1].replace(".pcd",".jpg")
            for img_file in annos['images']:
                if img_file['file_name']==fn:
                    img_id = img_file['id']
            masks, ids = [],[]
            for img_ann in annos['annotations']:
                if img_ann['image_id']==img_id:
                    rle = pycoco_mask.frPyObjects(img_ann['segmentation'], 480, 848 )
                    masks.append(pycoco_mask.decode(rle).squeeze(-1))
                    ids.append(img_ann['attributes']['track_id'])

            pcd = o3d.io.read_point_cloud(pcd_file)
            image = cv2.imread(img_fn,0)
            if masks:
                masks = np.array(masks)[:,200:,:720]
            #masks = cv2.imread(mask_file,0)[200:,:720]
            print(ds_pth,i, ids)
            annotation = []
            for inst,track_id in zip(masks,ids):
                inst_pcd = crop_with_2dmask(pcd,inst)
                cl, ind = inst_pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=1.0)
                pcl_processed = inst_pcd.select_by_index(ind)
                if np.array(pcl_processed.points).shape[0] > 20:
                    box = pcl_processed.get_axis_aligned_bounding_box()

                    # x-axis
                    left = [1, 0, 0]
                    # y-axis
                    front = [0, 1, 0]
                    # z-axis
                    up = [0, 0, 1]
                    size = np.array(box.get_extent())
                    yaw=0.
                    center = np.array(box.get_center())
                    #################################################
                    size[1], size[0] = size[0], size[1]
                    annotation.append(
                        Cuboid3d(position=center,
                                 scale=list(size),
                                 rotation=[0,0,-1.5708],
                                 label=0,
                                 attributes={'track_id': track_id+1})
                    )

                    #"orientation": dict(rotationYaw=0,rotationPitch=0,rotationRoll=270)

                    ann_id+=0
            data_item = DatasetItem(id=pcd_file.split("/")[-1].replace(".pcd",""),
                                    annotations=annotation,
                                    point_cloud=pcd,
                                    related_images=[image, fn],
                                    attributes={'frame': i})
            dataset_items.append(data_item)

            #with open(f'{pcd_file.replace("point_cloud", "annotations").replace(".pcd",".json")}',"w") as f:
            #    json.dump(annos_out,f)
        dataset = Dataset.from_iterable(dataset_items, categories=['goat'])
        dataset.export(f'{args.root_dir.replace("2d","3d")}/{ds_pth}', format='sly_pointcloud', save_images=False)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str , default="/media/mlk/storage/AMMOD/ammod_realsense/data/2d/train", help="path to directory containing video clips"
    )
    args = parser.parse_args()
    main(args)