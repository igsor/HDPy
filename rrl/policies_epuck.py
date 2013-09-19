"""

.. todo::
    documentation

"""
from rl import Policy
import warnings
import numpy as np

class Heading(Policy):
    """
    """
    def __init__(self):
        super(Heading, self).__init__(action_space_dim=1)
        self.action = self.initial_action()
    
    def initial_action(self):
        """
        """
        return np.atleast_2d([0.0]).T
    
    def update(self, action_upd):
        """
        """
        self.action = action_upd
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        """
        """
        return self.action
    
    def reset(self):
        self.action = self.initial_action()

class HeadingRandInit(Heading):
    """
    """
    def initial_action(self):
        """
        """
        rnd = np.random.uniform(0.0, 2*np.pi)
        return np.atleast_2d([rnd]).T

class Trivial(Heading):
    """
    """
    def __init__(self):
        warnings.warn("This class is deprecated. Use 'Heading' instead")
        super(Trivial, self).__init__()

class RandInit(HeadingRandInit):
    """
    """
    def __init__(self):
        warnings.warn("This class is deprecated. Use 'HeadingRandInit' instead")
        super(RandInit, self).__init__()
