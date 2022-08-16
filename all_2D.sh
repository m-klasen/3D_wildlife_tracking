python main.py --mode 2D --iou=0.01 --root_dir /media/mlk/storage/AMMOD/ammod_realsense/data/2d/train
python main.py --mode 2D --iou=0.01 --root_dir /media/mlk/storage/AMMOD/ammod_realsense/data/2d/valid

python main.py --mode 2D --iou=0.25 --root_dir /media/mlk/storage/AMMOD/ammod_realsense/data/2d/train
python main.py --mode 2D --iou=0.25 --root_dir /media/mlk/storage/AMMOD/ammod_realsense/data/2d/valid

python main.py --mode 2D --iou=0.5 --root_dir /media/mlk/storage/AMMOD/ammod_realsense/data/2d/train
python main.py --mode 2D --iou=0.5 --root_dir /media/mlk/storage/AMMOD/ammod_realsense/data/2d/valid

python evaluation/evaluate_3dmot.py 2D/inst_segm_depth_median_4_aabbox_kalman_iou0.01
python evaluation/evaluate_3dmot.py 2D/inst_segm_depth_median_4_aabbox_kalman_iou0.25
python evaluation/evaluate_3dmot.py 2D/inst_segm_depth_median_4_aabbox_kalman_iou0.5