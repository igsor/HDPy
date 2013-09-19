"""

.. todo::
    documentation

"""
from rl import Plant
import warnings
import numpy as np

class CollisionAvoidanceFrontal(Plant):
    """
    """
    def __init__(self, theta, obs_noise=0.0):
        super(CollisionAvoidanceFrontal, self).__init__(state_space_dim=3)
        self.theta = float(theta)
        self.obs_sigma = abs(float(obs_noise))
    
    def state_input(self, state):
        """Return the state from observations ``state``"""
        input_ = np.hstack((state['ir'][0, :2], [state['ir'][0, -1]]))
        input_ = self.normalization.normalize_value('ir', input_)
        #input_ += np.random.normal(scale=0.001, size=input_.shape) # Additive noise
        return np.atleast_2d(input_).T
    
    def reward(self, epoch):
        """Return the reward produced by ``epoch``."""
        ir_front = np.hstack((epoch['ir'][:2], [epoch['ir'][-1]]))
        ret = float(sum([min(ir - self.theta, 0) for ir in ir_front.T]))
        #ret += np.random.normal(scale=0.00001)
        if self.obs_sigma > 0.0:
            ret += np.random.normal(scale=self.obs_sigma)
        return ret

class CollisionAvoidanceSideways(Plant):
    """
    """
    def __init__(self, theta, obs_noise=0.0):
        super(CollisionAvoidanceSideways, self).__init__(state_space_dim=3)
        self.theta = float(theta)
        self.obs_sigma = abs(float(obs_noise))
    
    def state_input(self, state):
        """Return the state from observations ``state``"""
        input_ = np.array((state['ir'][0, 0], state['ir'][0, 2], state['ir'][0, 6]))
        input_ = self.normalization.normalize_value('ir', input_)
        #input_ += np.random.normal(scale=0.001, size=input_.shape) # Additive noise
        return np.atleast_2d(input_).T
    
    def reward(self, epoch):
        """Return the reward produced by ``epoch``."""
        sensors = np.array((epoch['ir'][0, 0], epoch['ir'][0, 2], epoch['ir'][0, 6]))
        ret = float(sum([min(ir - self.theta, 0) for ir in sensors]))
        if self.obs_sigma > 0.0:
            ret += np.random.normal(scale=self.obs_sigma)
        return ret

class CollisionAvoidanceFull(Plant):
    """
    """
    def __init__(self, theta, obs_noise=0.0):
        super(CollisionAvoidanceFull, self).__init__(state_space_dim=8)
        self.theta = float(theta)
        self.obs_sigma = abs(float(obs_noise))
    
    def state_input(self, state):
        """Return the state from observations ``state``"""
        input_ = state['ir'].T
        input_ = self.normalization.normalize_value('ir', input_)
        return input_
    
    def reward(self, epoch):
        """Return the reward produced by ``epoch``."""
        ret = float(sum([min(ir - self.theta, 0) for ir in epoch['ir'].T]))
        #ret += np.random.normal(scale=0.00001)
        if self.obs_sigma > 0.0:
            ret += np.random.normal(scale=self.obs_sigma)
        return ret

class Location(Plant):
    """
    """
    def __init__(self, theta):
        super(Location, self).__init__(state_space_dim=3)
        self.theta = float(theta)
    
    def state_input(self, state):
        """Return the state from observations ``state``"""
        input_ = np.hstack((state['loc'], state['pose'])).T
        return input_
    
    def reward(self, epoch):
        """Return the reward produced by ``epoch``."""
        ret = float(sum([min(ir - self.theta, 0) for ir in epoch['ir'].T]))
        #ret += np.random.normal(scale=0.00001)
        return ret

class Attractor(Plant):
    """
    """
    def __init__(self, attractor, repeller, scale):
        self.attractor = map(float, attractor)
        self.repeller = map(float, repeller)
        self.scale = scale
        super(Attractor, self).__init__(state_space_dim=2)
    
    def state_input(self, state):
        """Return the state from observations ``state``"""
        input_ = np.atleast_2d(state['mdist']).T
        return input_
    
    def _idist(self, pt0, pt1):
        """Compute the inverse distance between two points ``pt0`` and
        ``pt1``. The points are expected to be coordinate tuples.
        """
        x_0, y_0 = pt0
        x_1, y_1 = pt1
        return 1.0 / np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
    
    def reward(self, epoch):
        """Return the reward produced by ``epoch``."""
        reward = 0.0
        reward += self.scale * self._idist(epoch['loc'][0], self.attractor)
        reward -= self.scale * self._idist(epoch['loc'][0], self.repeller)
        return reward


class Trivial(CollisionAvoidanceFrontal):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`CollisionAvoidanceFrontal` instead
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("This class is deprecated. Use 'CollisionAvoidance' instead")
        super(Trivial, self).__init__(*args, **kwargs)

class SidewaysTrivial(CollisionAvoidanceSideways):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`CollisionAvoidanceFrontal` instead
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("This class is deprecated. Use 'CollisionAvoidanceSideways' instead")
        super(SidewaysTrivial, self).__init__(*args, **kwargs)

class FullTrivial(CollisionAvoidanceFull):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`CollisionAvoidanceFrontal` instead
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("This class is deprecated. Use 'CollisionAvoidanceFull' instead")
        super(FullTrivial, self).__init__(*args, **kwargs)
