import os, numpy as np, sys, cv2
import glob
from PIL import Image
from PIL import ImageEnhance
from xinshuo_io import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts
from xinshuo_visualization import random_colors
from kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

max_color = 10
colors = random_colors(max_color)       # Generate random colors
type_whitelist = ['goat']
score_threshold = -10000
width = 720
height = 280
seq_list = ['20201217170333']

def color_map_color(value, cmap_name='tab20', vmin=0, vmax=20):
    value = value%vmax
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

def vis(result_sha, data_root, result_root):
	def show_image_with_boxes(img, pcl, objects_res, object_gt, calib, save_path, height_threshold=0):
		img2 = np.copy(img) 
		b = 80
		h,w,c  = img.shape
		new_out = np.zeros((b+h+b, b+w+b, c), dtype=np.uint8); new_out.fill(255)
		new_out[b:-b, b:-b] = img2

		plt.figure(figsize=(16,10))

		
		plt.imshow(new_out)
		#plt.tight_layout()
		ax = plt.gca()
		#ax = None
		for obj in objects_res:
			box3d_pts_2d, _ = compute_box_3d(pcl, obj, calib)
			box3d_pts_2d += b

			color_tmp = color_map_color(obj.id, vmax=max_color)#tuple([tmp for tmp in colors[obj.id % max_color]])
			# if obj.id==1:
			# 	color_tmp = matplotlib.colors.to_rgb('red')
			# if obj.id==3:
			# 	color_tmp = matplotlib.colors.to_rgb('blue')
			# if obj.id==4:
			# 	color_tmp = matplotlib.colors.to_rgb('green')
			# if obj.id==5:
			# 	color_tmp = matplotlib.colors.to_rgb('orange')
			new_out, ax = draw_projected_box3d(new_out, ax, box3d_pts_2d, color=color_tmp)
			text = 'ID: %d' % obj.id
			if box3d_pts_2d is not None:
				#img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]) - 4), 
                #       				cv2.FONT_HERSHEY_TRIPLEX, 0.6, color=color_tmp, lineType=cv2.LINE_AA)
				ax.text(int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]), text, fontsize=15,
						bbox=dict(facecolor='yellow', alpha=0.5))
		#plt.show()
		plt.axis('off')
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()
		#img = Image.fromarray(img2)
		#img.save(save_path)
		return np.array(new_out)
	
	for seq in seq_list:
		image_dir = os.path.join(data_root, '%s/color/' % seq)
		point_dir = os.path.join(data_root, '%s/pointcloud_dpt/' % seq)
		calib_file = os.path.join(data_root, '%s/calib/000001_left.txt' % seq)
		result_dir = os.path.join(result_root, '%s/trk_withid/%s' % (result_sha, seq))
		save_3d_bbox_dir = os.path.join(result_dir); mkdir_if_missing(save_3d_bbox_dir)

		# load the list
		images_list = sorted(os.listdir(image_dir))
		pcloud_list = sorted(os.listdir(point_dir))
		num_images = len(images_list)
		print('number of images to visualize is %d' % num_images)
		start_count = 0
		out = cv2.VideoWriter(f'{seq}.avi', cv2.VideoWriter_fourcc(*'XVID'), 15.0, (848,480))
		for count in range(start_count, num_images):
			image_fn = images_list[count]
			pcl_fn   = pcloud_list[count]
			pcl = o3d.io.read_point_cloud(point_dir+pcl_fn)
   
			image_tmp = Image.open(image_dir+image_fn)
			enhancer = ImageEnhance.Brightness(image_tmp)
			enhanced_im = enhancer.enhance(1.3)
			enhancer = ImageEnhance.Contrast(enhanced_im)
			enhanced_im = enhancer.enhance(1.3)
			image_tmp = np.array(enhanced_im)
			1
			image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_GRAY2BGR )

			result_tmp = os.path.join(result_dir, image_fn.replace(".jpg",".txt"))		# load the result
			if not is_path_exists(result_tmp): object_res = []
			else: object_res = read_label(result_tmp)
			print('processing index: %s, %d/%d, results from %s' % (image_fn, count+1, num_images, result_tmp))
			calib_tmp = Calibration(calib_file)			# load the calibration

			object_res_filtered = []
			for object_tmp in object_res:
				if object_tmp.type not in type_whitelist: continue
				if hasattr(object_tmp, 'score'):
					if object_tmp.score < score_threshold: continue
				center = object_tmp.t
				object_res_filtered.append(object_tmp)

			num_instances = len(object_res_filtered)
			save_image_with_3dbbox_gt_path = os.path.join(save_3d_bbox_dir, image_fn)
			img = show_image_with_boxes(image_tmp, pcl, object_res_filtered, [], calib_tmp, save_path=save_image_with_3dbbox_gt_path)
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR )
			out.write(img)
			print('number of objects to plot is %d' % (num_instances))
			count += 1
		out.release()
		#cv2.destroyAllWindows()

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print('Usage: python visualization.py result_sha(e.g., pointrcnn_Car_test_thres)')
		sys.exit(1)

	result_root = 'results'
	result_sha = sys.argv[1]

	data_root = '/media/mlk/storage/data/train/ldth'

	#vis(result_sha, data_root, result_root)

	out = cv2.VideoWriter(f'20201217170333.avi', cv2.VideoWriter_fourcc(*'XVID'), 15.0, (1232,790))
	for img_fn in sorted(glob.glob('results/3D/iou0.01/trk_withid/20201217170333/*.jpg')):
		img = cv2.imread(img_fn)
		print(img_fn, img.shape)
		out.write(img)
	out.release()