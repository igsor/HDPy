from ...rl import Plant
import numpy as np
import warnings

class SpeedReward(Plant):
    """A :py:class:`Plant` with focus on the speed of the robot.
    """
    def __init__(self):
        super(SpeedReward, self).__init__(state_space_dim=2)
    
    def state_input(self, state):
        """Return the location, sampled from the *GPS* (x,y) values.
        The sample is an average over the last 10 GPS coordinates.
        """
        sio =  np.atleast_2d([
            self.normalization.normalize_value('puppyGPS_x', state['puppyGPS_x'][-10:]).mean(),
            self.normalization.normalize_value('puppyGPS_y', state['puppyGPS_y'][-10:]).mean()
        ]).T
        return sio
    
    def reward(self, epoch):
        """Return the covered distance and -1.0 if the robot tumbled.
        The speed measurement is taken from the 100th to the last sample.
        """
        if (epoch['accelerometer_z'][-100:] < 1.0).sum() > 80:
            return -1.0
        
        x = epoch['puppyGPS_x']
        y = epoch['puppyGPS_y']
        return np.linalg.norm(np.array([x[-1] - x[-100], y[-1] - y[-100]]))

class LineFollower(Plant):
    """A :py:class:`Plant` which gives negative reward proportional to
    the distance to a line in the xy plane. The line is described by
    its ``origin`` and the ``direction``.
    """
    def __init__(self, origin, direction, reward_noise=0.01):
        super(LineFollower, self).__init__(state_space_dim=2)
        self.origin = np.atleast_2d(origin)
        self.direction = np.atleast_2d(direction)
        self.reward_noise = reward_noise
        
        if self.origin.shape[0] < self.origin.shape[1]:
            self.origin = self.origin.T
            
        if self.direction.shape[0] < self.direction.shape[1]:
            self.direction = self.direction.T
        
        self.direction /= np.linalg.norm(self.direction)
        
        assert self.direction.shape == (2, 1)
        assert self.origin.shape == (2, 1)
    
    def state_input(self, state):
        """Return the latest *GPS* (x,y) values.
        """
        sio =  np.atleast_2d([
            self.normalization.normalize_value('puppyGPS_x', state['puppyGPS_x'][-10:]).mean(),
            self.normalization.normalize_value('puppyGPS_y', state['puppyGPS_y'][-10:]).mean()
        ]).T
        return sio
    
    def reward(self, epoch):
        """Return the distance between the current robot location and
        the line.
        """
        x = epoch['puppyGPS_x'][-1]
        y = epoch['puppyGPS_y'][-1]
        point = np.atleast_2d([x, y]).T
        
        #(origin - point) - (<origin - point, dir>) * dir
        diff = self.origin - point
        proj = diff - self.direction.T.dot(diff).dot(self.direction.T).T
        #return np.tanh(1.0/np.linalg.norm(proj))
        
        reward = -np.linalg.norm(proj)
        reward += np.random.normal(scale=self.reward_noise, size=reward.shape)
        return reward

class TargetLocation(Plant):
    """A :py:class:`Plant` which gives negative reward proportional to
    the distance to point ``target`` in the xy plane. If the robot is
    closer than ``radius`` to the target, the reward will be 0.0.
    
    """
    def __init__(self, target, radius=0.0, reward_noise=0.01):
        super(TargetLocation, self).__init__(state_space_dim=2)
        self.target = np.atleast_2d(target)
        self.radius = radius
        self.reward_noise = reward_noise
        
        if self.target.shape[0] < self.target.shape[1]:
            self.target = self.target.T
            
        assert self.target.shape == (2, 1)
    
    def state_input(self, state):
        """Return the latest *GPS* (x,y) values."""
        sio =  np.atleast_2d([
            self.normalization.normalize_value('puppyGPS_x', state['puppyGPS_x'][-10:]).mean(),
            self.normalization.normalize_value('puppyGPS_y', state['puppyGPS_y'][-10:]).mean()
        ]).T
        return sio
    
    def reward(self, epoch):
        """Return the distance between the current robot location and
        the target point.
        
        """
        x = epoch['puppyGPS_x'][-1]
        y = epoch['puppyGPS_y'][-1]
        point = np.atleast_2d([x, y]).T
        
        
        #(target - point)
        diff = self.target - point
        dist = np.linalg.norm(diff)
        
        if dist < self.radius:
            dist = 0.0
        
        reward = np.exp(-0.25 * (dist - 9.0)) + 1.0
        
        if self.reward_noise > 0.0:
            reward += np.random.normal(scale=self.reward_noise)
        
        return reward

class TargetLocationLandmarks(TargetLocation):
    """A :py:class:`Plant` which gives negative reward proportional to
    the distance to point ``target`` in the xy plane. If the robot is
    closer than ``radius`` to the target, the reward will be 0.0.
    The state is composed of the distance to predefined ``landmarks``,
    specified with their coordinates in the xy plane. Gaussian noise
    will be added to the reward, if ``reward_noise`` is positive.
    
    """
    def __init__(self, target, landmarks, radius=0.0, reward_noise=0.01):
        super(TargetLocationLandmarks, self).__init__(target, radius, reward_noise)
        self._state_space_dim = len(landmarks)
        
        # add landmarks
        self.landmarks = []
        for mark in landmarks:
            mark = np.atleast_2d(mark)
            if mark.shape[0] < mark.shape[1]:
                mark = mark.T
            self.landmarks.append(mark)
    
    def state_input(self, state):
        """Return the distance to the landmarks."""
        sio =  np.atleast_2d([
            state['puppyGPS_x'][-10:].mean(),
            state['puppyGPS_y'][-10:].mean()
        ]).T
        
        dist = [np.linalg.norm(sio - mark) for mark in self.landmarks]
        dist = np.atleast_2d(dist).T
        dist = self.normalization.normalize_value('landmark_dist', dist)
        return dist

class DiffTargetLocationLandmarks(TargetLocationLandmarks):
    """A :py:class:`Plant` which gives positive reward proportional to
    the absolute difference  (between two episodes) in distance to
    point ``target`` in the xy plane. The state is composed of the
    distance to predefined ``landmarks``,
    specified with their coordinates in the xy plane. Gaussian noise
    will be added to the reward, if ``reward_noise`` is positive.
    
    Before the first call, the distance is set to ``init_distance``.
    
    """
    def __init__(self, target, landmarks, reward_noise=0.01, init_distance=100):
        super(DiffTargetLocationLandmarks, self).__init__(target, landmarks, 0.0, reward_noise)
        self.init_distance = init_distance
        self._last_target_distance = self.init_distance # TODO: what is good init value?
    
    def reward(self, epoch):
        """Return the reward of ``epoch``."""
        x = epoch['puppyGPS_x'][-1]
        y = epoch['puppyGPS_y'][-1]
        point = np.atleast_2d([x, y]).T
        
        
        #(target - point)
        diff = self.target - point
        dist = np.linalg.norm(diff)
        
        # reward is difference of distance between current and previous episode
        reward = dist - self._last_target_distance
        self._last_target_distance = dist
        reward += np.random.normal(scale=self.reward_noise, size=reward.shape)
        return reward
    
    def reset(self):
        """Reset the last distance to the initial one."""
        self._last_target_distance = self.init_distance

class LandmarksTarLoc(TargetLocationLandmarks):
    """A :py:class:`Plant` which gives negative reward proportional to
    the distance to point ``target`` in the xy plane.
    
    .. deprecated:: 1.0
        Use :py:class:`TargetLocationLandmarks` instead.
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn('This class is depcreated. Use TargetLocationLandmarks instead')
        super(LandmarksTarLoc, self).__init__(*args, **kwargs)

class LandmarksTarLocDiff(DiffTargetLocationLandmarks):
    """A :py:class:`Plant` which gives positive reward proportional to
    the absolute difference  (between two episodes) in distance to
    point ``target`` in the xy plane.
    
    .. deprecated:: 1.0
        Use :py:class:`DiffTargetLocationLandmarks` instead.
    
    """
    def __init__(self, *args, **kwargs):
        warnings.warn('This class is depcreated. Use DiffTargetLocationLandmarks instead')
        super(LandmarksTarLocDiff, self).__init__(*args, **kwargs)
