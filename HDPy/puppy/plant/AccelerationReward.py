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
        super(AccelerationReward, self).__init__(state_space_dim=24)
        self.x = []
        self.y = []
        self.ax = []
        self.ay = []
        self.az = []


    def state_input(self, state):
        """Full state
        """
        sio =  np.atleast_2d([
            self.normalization.normalize_value('puppyGPS_x', state['puppyGPS_x'][-1]),
            self.normalization.normalize_value('puppyGPS_y', state['puppyGPS_y'][-1]),
            self.normalization.normalize_value('puppyGPS_z', state['puppyGPS_z'][-1]),
            self.normalization.normalize_value('accelerometer_x', state['accelerometer_x'][-1]),
            self.normalization.normalize_value('accelerometer_y', state['accelerometer_y'][-1]),
            self.normalization.normalize_value('accelerometer_z', state['accelerometer_z'][-1]),
            self.normalization.normalize_value('compass_x', state['compass_x'][-1]),
            self.normalization.normalize_value('compass_y', state['compass_y'][-1]),
            self.normalization.normalize_value('compass_z', state['compass_z'][-1]),
            self.normalization.normalize_value('gyro_x', state['gyro_x'][-1]),
            self.normalization.normalize_value('gyro_y', state['gyro_y'][-1]),
            self.normalization.normalize_value('gyro_z', state['gyro_z'][-1]),
            self.normalization.normalize_value('hip0', state['hip0'][-1]),
            self.normalization.normalize_value('hip1', state['hip1'][-1]),
            self.normalization.normalize_value('hip2', state['hip2'][-1]),
            self.normalization.normalize_value('hip3', state['hip3'][-1]),
            self.normalization.normalize_value('knee0', state['knee0'][-1]),
            self.normalization.normalize_value('knee1', state['knee1'][-1]),
            self.normalization.normalize_value('knee2', state['knee2'][-1]),
            self.normalization.normalize_value('knee3', state['knee3'][-1]),
#             state['touch0'][-1],
#             state['touch1'][-1],
#             state['touch2'][-1],
#             state['touch3'][-1],
            self.normalization.normalize_value('touch0', state['touch0'][-1]),
            self.normalization.normalize_value('touch1', state['touch1'][-1]),
            self.normalization.normalize_value('touch2', state['touch2'][-1]),
            self.normalization.normalize_value('touch3', state['touch3'][-1])
        ]).T
        return sio


    def reward(self, epoch):
        """Return -100.0 if the robot tumbled.
        Maximizes speed while minimizing total acceleration
        The speed measurement is the average/covered distance since last epoch
        and sum of the acceleration minus gravity is used as negative reinforcement.
        """
        
#         if (epoch['accelerometer_z'] < 1.0).mean() > 0.8:
#             return -100.0
        
        n = epoch['puppyGPS_x'].size
        
        #keep last position
        self.x  = np.concatenate([self.x[-1:],  epoch['puppyGPS_x']])
        self.y  = np.concatenate([self.y[-1:],  epoch['puppyGPS_y']])
        
        #store last 2 epochs plus current one 
        self.ax = np.concatenate([self.ax[-2*n:], epoch['accelerometer_x']])
        self.ay = np.concatenate([self.ay[-2*n:], epoch['accelerometer_y']])
        self.az = np.concatenate([self.az[-2*n:], epoch['accelerometer_z']])
        
        spd = 0
        if self.x.size > 1:
            mov = np.linalg.norm(np.array([self.x[-1] - self.x[0], self.y[-1] - self.y[0]]))
            #check consistency
            if mov < 0.1*n:
                # calculate displacement in a reasonable scale
                spd = (3000.0/n) * mov;
        
        
        
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
