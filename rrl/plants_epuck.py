
import rrl
import numpy as np
from math import sqrt

class Trivial(rrl.Plant):
    def __init__(self, theta, obs_noise=0.0):
        super(Trivial, self).__init__(state_space_dim=3)
        self.theta = float(theta)
        self.obs_sigma = abs(float(obs_noise))
    
    def state_input(self, state):
        input_ = np.hstack((state['ir'][0,:2], [state['ir'][0,-1]]))
        input_ = self.normalization.normalize_value('ir', input_)
        #input_ += np.random.normal(scale=0.001, size=input_.shape) # Additive noise
        return np.atleast_2d(input_).T
    
    def reward(self, epoch):
        ir_front = np.hstack((epoch['ir'][:2], [epoch['ir'][-1]]))
        ret = float(sum([min(ir - self.theta, 0) for ir in ir_front.T]))
        #ret += np.random.normal(scale=0.00001)
        if self.obs_sigma > 0.0:
            ret += np.random.normal(scale=self.obs_sigma)
        return ret

class SidewaysTrivial(rrl.Plant):
    def __init__(self, theta, obs_noise=0.0):
        super(SidewaysTrivial, self).__init__(state_space_dim=3)
        self.theta = float(theta)
        self.obs_sigma = abs(float(obs_noise))
    
    def state_input(self, state):
        input_ = np.array((state['ir'][0,0], state['ir'][0,2], state['ir'][0,6]))
        input_ = self.normalization.normalize_value('ir', input_)
        #input_ += np.random.normal(scale=0.001, size=input_.shape) # Additive noise
        return np.atleast_2d(input_).T
    
    def reward(self, epoch):
        sensors = np.array((epoch['ir'][0,0], epoch['ir'][0,2], epoch['ir'][0,6]))
        ret = float(sum([min(ir - self.theta, 0) for ir in sensors]))
        if self.obs_sigma > 0.0:
            ret += np.random.normal(scale=self.obs_sigma)
        return ret

class FullTrivial(rrl.Plant):
    def __init__(self, theta, obs_noise=0.0):
        super(FullTrivial, self).__init__(state_space_dim=8)
        self.theta = float(theta)
        self.obs_sigma = abs(float(obs_noise))
    
    def state_input(self, state):
        input_ = state['ir'].T
        input_ = self.normalization.normalize_value('ir', input_)
        return input_
    
    def reward(self, epoch):
        ret = float(sum([min(ir - self.theta, 0) for ir in epoch['ir'].T]))
        #ret += np.random.normal(scale=0.00001)
        if self.obs_sigma > 0.0:
            ret += np.random.normal(scale=self.obs_sigma)
        return ret

class Location(rrl.Plant):
    def __init__(self, theta):
        super(Location, self).__init__(state_space_dim=3)
        self.theta = float(theta)
    
    def state_input(self, state, action):
        input_ = np.hstack((state['loc'], state['pose'])).T
        return input_
    
    def reward(self, epoch):
        ret = float(sum([min(ir - self.theta, 0) for ir in epoch['ir'].T]))
        #ret += np.random.normal(scale=0.00001)
        return ret

class Attractor(rrl.Plant):
    def __init__(self, attractor, repeller, scale):
        self.attractor = map(float, attractor)
        self.repeller = map(float, repeller)
        self.scale = scale
        super(Attractor, self).__init__(state_space_dim=2)
    
    def state_input(self, state, action):
        input_ = np.atleast_2d(state['mdist']).T
        return input_
    
    def _idist(self, pt0, pt1):
        x0,y0 = pt0
        x1,y1 = pt1
        return 1.0/sqrt((x0-x1)**2 + (y0-y1)**2)
    
    def reward(self, epoch):
        reward = self.scale * self._idist(epoch['loc'][0], self.attractor)
        reward -= self.scale * self._idist(epoch['loc'][0], self.repeller)
        return reward
