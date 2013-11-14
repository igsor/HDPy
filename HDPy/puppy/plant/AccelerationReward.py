"""
ACD plants


"""
from HDPy import Plant
import numpy as np
import scipy.constants
import scipy.signal

class AccelerationReward(Plant):
    """A :py:class:`Plant` with focus on the speed and acceleration of the robot.
    """

    def __init__(self):
        super(AccelerationReward, self).__init__(state_space_dim=28)
        self.ax = []
        self.ay = []
        self.az = []


    def state_input(self, state):
        """Full state
        """
        return np.atleast_2d(state.values())


    def reward(self, epoch):
        """Return -100.0 if the robot tumbled.
        Maximizes speed while minimizing total acceleration
        The speed measurement is the average/covered distance since last epoch
        and sum of the acceleration minus gravity is used as negative reinforcement.
        """
        
        if (epoch['accelerometer_z'] < 1.0).mean() > 0.8:
            return -100.0
        
        x = epoch['puppyGPS_x']
        y = epoch['puppyGPS_y']
        n = x.size
        
        # calculate displacement in a reasonable scale
        spd = (3000.0/n) * np.linalg.norm(np.array([x[-1] - x[0], y[-1] - y[0]]));
        
        
        #store last 2 epochs plus current one 
        self.ax = np.concatenate([self.ax[-2*n:], epoch['accelerometer_x']])
        self.ay = np.concatenate([self.ay[-2*n:], epoch['accelerometer_y']])
        self.az = np.concatenate([self.az[-2*n:], epoch['accelerometer_z']])
        
        s = np.ceil(self.ax.size/3.0)
        fr = 0.3
        sr = 2*fr + (s/10.0) #should be smaller than s
        
        #filtered to remove noise; borders of the result always tend to zero and have to be trimmed
        end = -np.ceil(sr)
        beg = -s+end
        fax = firfilt(self.ax, fr, sr)[beg:end]
        fay = firfilt(self.ay, fr, sr)[beg:end]
        faz = firfilt(self.az, fr, sr)[beg:end]
        
        if fax.size > 0:
            acc = abs(fax  + fay  + faz - scipy.constants.g).mean()
        else:
            acc = scipy.constants.g;

        #acc = abs(epoch['accelerometer_x'] + epoch['accelerometer_y']  + epoch['accelerometer_z'] - scipy.constants.g).mean()
        return spd - acc; 


def firfilt(interval, freq, sampling_rate):
    """ Second Order LowPass Filter
    """
    nfreq = freq/(0.5*sampling_rate)
    taps =  sampling_rate + 1
    a = 1
    b = scipy.signal.firwin(taps, cutoff=nfreq)
    firstpass = scipy.signal.lfilter(b, a, interval)
    secondpass = scipy.signal.lfilter(b, a, firstpass[::-1])[::-1]
    return secondpass
