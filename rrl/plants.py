"""
ACD plants


"""
from rl import Plant
import numpy as np

class SpeedReward(Plant):
    """A :py:class:`Plant` with focus on the speed of the robot.
    """
    def __init__(self):
        super(SpeedReward, self).__init__(state_space_dim=2)
    
    def state_input(self, state):
        """Return the latest *GPS* (x,y) values.
        """
        sio =  np.atleast_2d([
            state['puppyGPS_x'][-1],
            state['puppyGPS_y'][-1]
        ]).T
        return sio
    
    def reward(self, epoch):
        """Return the covered distance and -1.0 if the robot tumbled.
        """
        if (epoch['accelerometer_z'] < 1.0).sum() > 80:
            return -1.0
        
        x = epoch['puppyGPS_x']
        y = epoch['puppyGPS_y']
        return np.linalg.norm(np.array([x[-1] - x[0], y[-1] - y[0]]))

class LineFollower(Plant):
    """A :py:class:`Plant` which gives negative reward proportional to
    the distance to a line in the xy plane. The line is described by
    its ``origin`` and the ``direction``.
    
    .. todo::
        Average over multiple samples of the epoch
    
    """
    def __init__(self, origin, direction):
        super(LineFollower, self).__init__(state_space_dim=2)
        self.origin = np.atleast_2d(origin)
        self.direction = np.atleast_2d(direction)
        
        if self.origin.shape[0] < self.origin.shape[1]:
            self.origin = self.origin.T
            
        if self.direction.shape[0] < self.direction.shape[1]:
            self.direction = self.direction.T
        
        self.direction /= np.linalg.norm(self.direction)
        
        assert self.direction.shape == (2,1)
        assert self.origin.shape == (2,1)
    
    def state_input(self, state):
        """Return the latest *GPS* (x,y) values.
        """
        sio =  np.atleast_2d([
            state['puppyGPS_x'][-1],
            state['puppyGPS_y'][-1]
        ]).T
        return sio
    
    def reward(self, epoch):
        """Return the distance between the current robot location and
        the line.
        """
        x = epoch['puppyGPS_x'][-1]
        y = epoch['puppyGPS_y'][-1]
        point = np.atleast_2d([x,y]).T
        
        #(origin - point) - (<origin - point, dir>) * dir
        diff = self.origin - point
        proj = diff - self.direction.T.dot(diff).dot(self.direction.T).T
        return -np.linalg.norm(proj)

class TargetLocation(Plant):
    """A :py:class:`Plant` which gives negative reward proportional to
    the distance to point ``target`` in the xy plane.
    
    .. todo::
        Average over multiple samples of the epoch
    """
    def __init__(self, target):
        super(TargetLocation, self).__init__(state_space_dim=2)
        self.target = np.atleast_2d(target)
        
        if self.target.shape[0] < self.target.shape[1]:
            self.target = self.target.T
            
        assert self.target.shape == (2,1)
    
    def state_input(self, state):
        """Return the latest *GPS* (x,y) values.
        """
        sio =  np.atleast_2d([
            state['puppyGPS_x'][-1],
            state['puppyGPS_y'][-1]
        ]).T
        return sio
    
    def reward(self, epoch):
        """Return the distance between the current robot location and
        the target point.
        """
        x = epoch['puppyGPS_x'][-1]
        y = epoch['puppyGPS_y'][-1]
        point = np.atleast_2d([x,y]).T
        
        #(target - point)
        diff = self.target - point
        return -np.linalg.norm(diff)


