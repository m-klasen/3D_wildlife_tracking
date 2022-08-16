"""
Modified Code from 
https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_particle_filter_object_tracking_video/ex_particle_filter_object_tracking_video.py
"""
import numpy as np
from numpy.random import uniform

class Filter(object):
    def __init__(self, bbox3D, info, ID):

        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1           		# number of total hits including the first detection
        self.info = info        		# other information associated	

class ParticleFilter(Filter): 
    def __init__(self, bbox3D, info, ID):
        super().__init__(bbox3D, info, ID)
        N=10000, std=0.5
        self.grid = [20,20,3]
        self.box_size = [1.5,1.5,2]
        self.dim=7
        self.std = std
        self.particles = np.empty((N, self.dim))
        self.particles[:, 0] = uniform(-self.grid[0], self.grid[0], size=N) #init the X coord
        self.particles[:, 1] = uniform(-self.grid[1], self.grid[1], size=N) #init the Y coord
        self.particles[:, 2] = uniform(-self.grid[2], self.grid[2], size=N) #init the Z coord
        self.particles[:, 3] = uniform(-3, 3, size=N) #init the theta coord
        self.particles[:, 4] = uniform(0, self.box_size[0], size=N) #init the w coord
        self.particles[:, 5] = uniform(0, self.box_size[1], size=N) #init the h coord
        self.particles[:, 6] = uniform(0, self.box_size[2], size=N) #init the l coord
        #Init the weiths vector as a uniform distribution
        #at the begining each particle has the same probability
        #to represent the point we are following
        #self.weights = np.empty((N, 1))
        self.weights = np.array([1.0/N]*N)
        #self.weights.fill(1.0/N) #normalised values

    def predict(self):

        x_velocity=0. 
        y_velocity=0.
        z_velocity=0.
        theta_velocity=0.
        w_velocity=0.
        h_velocity=0.
        l_velocity=0.
        self.particles[:, 0] += x_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the X coord
        self.particles[:, 1] += y_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the Y coord
        self.particles[:, 2] += z_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the X coord
        self.particles[:, 3] += theta_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the Y coord
        self.particles[:, 4] += w_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the X coord
        self.particles[:, 5] += l_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the Y coord
        self.particles[:, 6] += h_velocity + (np.random.randn(len(self.particles)) * self.std) #predict the X coord

    def update(self, bbox3d):
        x, y, z, theta, w,h,l = bbox3d
        #Generating a temporary array for the input position
        position = np.empty((len(self.particles), self.dim))
        position[:, 0].fill(x)
        position[:, 1].fill(y)
        position[:, 2].fill(z)
        position[:, 3].fill(theta)
        position[:, 4].fill(w)
        position[:, 5].fill(h)
        position[:, 6].fill(l)
        #1- We can take the difference between each particle new
        #position and the measurement. In this case is the Euclidean Distance.
        distance = np.linalg.norm(self.particles - position, axis=1)
        #2- Particles which are closer to the real position have smaller
        #Euclidean Distance, here we subtract the maximum distance in order
        #to get the opposite (particles close to the real position have
        #an higher wieght)
        max_distance = np.amax(distance)
        distance = np.add(-distance, max_distance)
        #3-Particles that best predict the measurement 
        #end up with the highest weight.
        self.weights.fill(1.0) #reset the weight array
        self.weights *= distance
        #4- after the multiplication the sum of the weights won't be 1. 
        #Renormalize by dividing all the weights by the sum of all the weights.
        self.weights += 1.e-300 #avoid zeros
        self.weights /= sum(self.weights) #normalize

    def estimate(self):

        x_mean = np.average(self.particles[:, 0], weights=self.weights, axis=0)
        y_mean = np.average(self.particles[:, 1], weights=self.weights, axis=0)
        z_mean = np.average(self.particles[:, 2], weights=self.weights, axis=0)
        theta_mean = np.average(self.particles[:, 3], weights=self.weights, axis=0)
        w_mean = np.average(self.particles[:, 4], weights=self.weights, axis=0)
        h_mean = np.average(self.particles[:, 5], weights=self.weights, axis=0)
        l_mean = np.average(self.particles[:, 6], weights=self.weights, axis=0)

        bbox3d = [x_mean,y_mean,z_mean,theta_mean,w_mean,h_mean,l_mean]
        return bbox3d
    
    def resample(self, method='residual'):

        N = len(self.particles)
        if(method == 'multinomal'):
            #np.cumsum() computes the cumulative sum of an array. 
            #Element one is the sum of elements zero and one, 
            #element two is the sum of elements zero, one and two, etc.
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1. #avoid round-off error
            #np.searchsorted() Find indices where elements should be 
            #inserted to maintain order. Here we generate random numbers 
            #in the range [0.0, 1.0] and do a search to find the weight 
            #that most closely matches that number. Large weights occupy 
            #more space than low weights, so they will be more likely 
            #to be selected.
            indices = np.searchsorted(cumulative_sum, np.random.uniform(low=0.0, high=1.0, size=N))      
        elif(method == 'residual'):
            indices = np.zeros(N, dtype=np.int32)
            # take int(N*w) copies of each weight
            num_copies = (N*np.asarray(self.weights)).astype(int)
            k = 0
            for i in range(N):
                for _ in range(num_copies[i]): # make n copies
                    indices[k] = i
                    k += 1
            #multinormial resample
            residual = self.weights - num_copies     # get fractional part
            residual /= sum(residual)     # normalize
            cumulative_sum = np.cumsum(residual)
            cumulative_sum[-1] = 1. # ensures sum is exactly one
            indices[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))
        elif(method == 'stratified'):
            #N subsets, chose a random position within each one
            #and generate a vector containing this positions
            positions = (np.random.random(N) + range(N)) / N
            #generate the empty indices vector
            indices = np.zeros(N, dtype=np.int32)
            #get the cumulative sum
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1
        elif(method == 'systematic'):
            # make N subsets, choose positions with a random offset
            positions = (np.arange(N) + np.random.random()) / N
            indices = np.zeros(N, dtype=np.int32)
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < N:
                if positions[i] < cumulative_sum[j]:
                    indices[i] = j
                    i += 1
                else:
                    j += 1
        else:
            raise ValueError("" + str(method) + "' is not implemented")
        #Create a new set of particles by randomly choosing particles 
        #from the current set according to their weights.
        self.particles[:] = self.particles[indices] #resample according to indices
        self.weights[:] = self.weights[indices]
        #Normalize the new set of particles
        self.weights /= np.sum(self.weights)        
