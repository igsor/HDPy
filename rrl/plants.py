"""
ACD plants


"""
from rl import Plant

class SpeedReward(Plant):
    """A :py:class:`Plant` with focus on the speed of the robot.
    """
    def __init__(self):
        super(SpeedReward, self).__init__(state_space_dim=2)
    
    def state_input(self, state, action):
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

