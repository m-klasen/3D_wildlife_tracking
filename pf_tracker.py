import numpy as np

from particle_filter import ParticleFilter

class ParticleBoxTracker(object):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self, bbox3D,info):
		"""
		Initialises a tracker using initial bounding box.
		"""
		# define constant velocity model
		self.pf = ParticleFilter()  


		self.time_since_update = 0
		self.id = ParticleBoxTracker.count
		ParticleBoxTracker.count += 1
		self.history = []
		self.hits = 1           # number of total hits including the first detection
		self.hit_streak = 1     # number of continuing hit considering the first detection
		self.first_continuing_hit = 1
		self.still_first = True
		self.age = 0
		self.info = info        # other info associated
  
		self.pf.predict()
		self.pf.update(bbox3D)
		self.pf.resample()
		
		
  
	def update(self, bbox3D,info): 
		""" 
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1          # number of continuing hit
		if self.still_first:
			self.first_continuing_hit += 1      # number of continuing hit in the fist time

		######################### orientation correction
		# if self.pf.particles[:,3] >= np.pi: self.pf.particles[:,3] -= np.pi * 2    # make the theta still in the range
		# if self.pf.particles[:,3] < -np.pi: self.pf.particles[:,3] += np.pi * 2

		# new_theta = bbox3D[3]
		# if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
		# if new_theta < -np.pi: new_theta += np.pi * 2
		# bbox3D[3] = new_theta

		# predicted_theta = self.kf.x[3]
		# if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
		# 	self.pf.particles[:,3] += np.pi       
		# 	if self.pf.particles[:,3] > np.pi: self.pf.particles[:,3] -= np.pi * 2    # make the theta still in the range
		# 	if self.pf.particles[:,3] < -np.pi: self.pf.particles[:,3] += np.pi * 2

		# # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		# if abs(new_theta - self.pf.particles[:,3]) >= np.pi * 3 / 2.0:
		# 	if new_theta > 0: self.pf.particles[:,3] += np.pi * 2
		# 	else: self.pf.particles[:,3] -= np.pi * 2

		#########################     # flip
		self.pf.update(bbox3D)
		self.pf.resample()
		
		#if self.pf.particles[:,3] >= np.pi: self.pf.particles[:,3] -= np.pi * 2    # make the theta still in the rage
		#if self.pf.particles[:,3] < -np.pi: self.pf.particles[:,3] += np.pi * 2

		self.info = info

	def predict(self):       
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		self.pf.predict()     
		# if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
		# if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		self.age += 1
		if (self.time_since_update > 0):
			self.hit_streak = 0
			self.still_first = False
		self.time_since_update += 1
		x = np.array(self.pf.estimate())
		self.history.append(x)
		return self.history[-1]

	def get_state(self):
		"""
		Returns the current bounding box estimate.
		"""
		return np.array(self.pf.estimate()).reshape((7, ))