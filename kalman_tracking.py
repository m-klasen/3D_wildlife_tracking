import numpy as np
from scipy.optimize import linear_sum_assignment
from evaluation.bbox_util import convert_3dbox_to_8corner, iou3d

from unscented_kalman_filter import UnscentedKalmanBoxTracker
from kalman_filter import KalmanBoxTracker

from pf_tracker import ParticleBoxTracker

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.01):   
	"""
	Assigns detections to tracked object (both represented as bounding boxes)
	detections:  N x 8 x 3
	trackers:	M x 8 x 3
	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	if (len(trackers)==0): 
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)	
	iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

	for d, det in enumerate(detections):
		for t, trk in enumerate(trackers):
			iou_matrix[d, t] = iou3d(det, trk)[0]			 # det: 8 x 3, trk: 8 x 3
	# matched_indices = linear_assignment(-iou_matrix)	  # hougarian algorithm, compatible to linear_assignment in sklearn.utils

	row_ind, col_ind = linear_sum_assignment(-iou_matrix)	  # hougarian algorithm
	matched_indices = np.stack((row_ind, col_ind), axis=1)

	unmatched_detections = []
	for d, det in enumerate(detections):
		if (d not in matched_indices[:, 0]): unmatched_detections.append(d)
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

	#filter out matched with low IOU
	matches = []
	for m in matched_indices:
		if (iou_matrix[m[0], m[1]] < iou_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else: matches.append(m.reshape(1, 2))
	if (len(matches) == 0): 
		matches = np.empty((0, 2),dtype=int)
	else: matches = np.concatenate(matches, axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Tracker(object):
	"""Tracker class that updates track vectors of object tracked
	Attributes:
		None
	"""

	def __init__(self, max_age=2, min_hits=3, iou=0.25):
		"""Initialize variable used by Tracker class
		Args:
			dist_thresh: distance threshold. When exceeds the threshold,
						 track will be deleted and new track is created
			max_frames_to_skip: maximum allowed frames to be skipped for
								the track object undetected
			max_trace_lenght: trace path history length
			trackIdCount: identification of each track object
		Return:
			None
		"""
		self.max_age = max_age
		self.min_hits = min_hits
		self.trackers = []
		self.frame_count = 0
		self.iou = iou
		# self.reorder = [3, 4, 5, 6, 2, 1, 0]
		# self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
		self.reorder = [0, 1, 2, 6, 5, 3, 4]
		self.reorder_back = [0, 1, 2, 5, 6, 4, 3]

	def update(self, dets_all):
		"""Update tracks vector using following steps:
			- Create tracks if no tracks vector found
			- Calculate cost using sum of square distance
			  between predicted vs detected centroids
			- Using Hungarian Algorithm assign the correct
			  detected measurements to predicted tracks
			  https://en.wikipedia.org/wiki/Hungarian_algorithm
			- Identify tracks with no assignment, if any
			- If tracks are not detected for long time, remove them
			- Now look for un_assigned detects
			- Start new tracks
			- Update KalmanFilter state, lastResults and tracks trace
		Args:
			detections: detected centroids of object to be tracked
		Return:
			None
		"""
		dets, info = dets_all['dets'], dets_all['info']
		dets = dets[:, self.reorder]
		info = info
  
		# remain_inds = (scores > 0.7).squeeze()
		# inds_low = scores > 0.1
		# inds_high = scores < 0.7
		# inds_second = np.logical_and(inds_low, inds_high).squeeze()
  
  
		# dets_second = dets[inds_second,:]
		# dets = dets[remain_inds,:]
		# scores_keep = scores[remain_inds]
		# scores_second = scores[inds_second]
  
  
		
		self.frame_count += 1

		trks = np.zeros((len(self.trackers), 7))		 # N x 7 , # get predicted locations from existing trackers.
		to_del = []
		ret = []
		for t, trk in enumerate(trks):
			pos = self.trackers[t].predict().reshape((-1, 1))
			trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]	   
			if (np.any(np.isnan(pos))): 
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
		for t in reversed(to_del): 
			self.trackers.pop(t)

		# First Assosiation
		dets_8corner = [convert_3dbox_to_8corner(det_tmp.squeeze()) for det_tmp in dets]
		if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		else: dets_8corner = []
		trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
		if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=self.iou)


		# update matched trackers with assigned detections
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]	 # a list of index
				trk.update(dets[d, :][0].squeeze(),info[d, :][0])
    
    
		# #Second Assosiation
		# dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_second]
		# if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		# else: dets_8corner = []
		# trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
		# if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
		# matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, iou_threshold=self.iou)


		# # update second matched trackers with assigned detections
		# for t, trk in enumerate(self.trackers):
		# 	if t not in unmatched_trks:
		# 		d = matched[np.where(matched[:, 1] == t)[0], 0]	 # a list of index
		# 		trk.update(dets_second[d, :][0],info[d, :][0])


		# create and initialise new trackers for unmatched detections
		for i in unmatched_dets:		# a scalar of index
			trk = KalmanBoxTracker(dets[i, :],info[i, :]) 
			#trk = ParticleBoxTracker(dets[i, :],info[i, :]) 
			self.trackers.append(trk)
		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()	  # bbox location
			d = d[self.reorder_back]			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]

			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):	  
				ret.append(np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1)) # +1 as MOT benchmark requires positive
			i -= 1

			# remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(i)
		if (len(ret) > 0): return np.concatenate(ret)			# h,w,l,x,y,z,theta, ID, other info, confidence
		return np.empty((0, 15))	