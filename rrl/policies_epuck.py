
import rrl
import numpy as np
from math import pi

class Trivial(rrl.Policy):
    def __init__(self):
        super(Trivial, self).__init__(action_space_dim=1)
        self.action = self.initial_action()
    
    def initial_action(self):
        return np.atleast_2d([0.0]).T
    
    def update(self, action_upd):
        self.action = action_upd
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        return self.action

class RandInit(rrl.Policy):
    def __init__(self):
        super(RandInit, self).__init__(action_space_dim=1)
        self.action = self.initial_action()
    
    def initial_action(self):
        rnd = np.random.uniform(0.0, 2*pi)
        return np.atleast_2d([rnd]).T
    
    def update(self, action_upd):
        self.action = action_upd
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        return self.action
