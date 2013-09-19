"""

"""
import numpy as np
from math import pi
import pylab
import rrl
import os
import PuPy

# ESN-ACD Libs
from environment import *
from epuck import Robot, AbsoluteRobot, OrientedRobot
import policy as policy_lib
import plant as plant_lib


## PARAMS ##

reservoir_dim = 150
theta = 1.0
gamma = 0.5
alpha = 0.3
#_max_step = 100
_max_episodes = 3000
momentum = 0.3

reservoir_pth = '/home/matthias/Desktop/esn_acd_reservoir.pic'
readout_pth = '/home/matthias/Desktop/esn_acd_readout.pic'
data_pth = '/home/matthias/Desktop/esn_acd.hdf5'

## INITIALIZATION ##

robot = AbsoluteRobot(
    obstacle_line=obstacles_box,
    obstacles=[train_lower, train_middle, train_left, train_upper],
    tol=0.0,
    speed=0.5,
    step_time=0.5,
)

policy = policy_lib.RandInit()
plant = plant_lib.Trivial(theta, obs_noise=0.05)

if reservoir_pth is not None and os.path.exists(reservoir_pth):
    print "Loading reservoir"
    import pickle
    f = open(reservoir_pth, 'r')
    reservoir = pickle.load(f)
    reservoir.reset()
    f.close()
    reservoir_dim = reservoir.get_output_dim()
else:
    reservoir = rrl.SparseReservoirNode(
        output_dim=reservoir_dim,
        input_dim=policy.action_space_dim() + plant.state_space_dim(),
        spectral_radius=0.95,
        reset_states=False,
        fan_in_i=100,
        input_scaling=1.0/3.0,
        bias_scaling=-3.0,
        fan_in_w=20
        )
    
    reservoir.initialize()
    reservoir.save(reservoir_pth + '.latest')

readout = rrl.StabilizedRLS(
    with_bias=True,
    input_dim=reservoir_dim+policy.action_space_dim() + plant.state_space_dim(),
    output_dim=1,
    lambda_=1.0
    )

if os.path.exists(data_pth):
    os.unlink(data_pth)


collided = False
class ExperimentingHDP(rrl.CollectingADHDP):
    def __init__(self, *args, **kwargs):
        import time
        t = time.time()
        seed = int((t - int(t)) * 10000.0)
        seed = 99045
        print "Seed =", seed
        np.random.seed(seed)
        super(ExperimentingHDP, self).__init__(*args, additional_headers={'seed':seed}, **kwargs)
    def _next_action_hook(self, a_next):
        """Postprocessing hook, after the next action ``a_next`` was
        proposed by the algorithm. Must return the possibly altered
        next action in the same format."""
        from math import pi
        
        # epsilon-greedy policy
        #if self.num_episode < 100:
        #    eps = 1.0
        #elif self.num_episode < 600:
        #    eps = 1.0 - 0.0016 * (self.num_episode - 100.0)
        #elif self.num_episode < 900:
        #    eps = 0.2 - 0.00066666666 * (self.num_episode - 600)
        #else:
        #    eps = 0.0
        
        #if np.random.uniform(0.0, 1.0) < eps:
        #    a_next = np.random.uniform(0.0, 2.0*pi, size=self.a_curr.shape)
        
        a_next = np.random.uniform(0.0, 2.0*pi, size=self.a_curr.shape)
        #a_next = np.random.uniform(-pi/2.0, pi/2.0, size=self.a_curr.shape)
        a_next = momentum * self.a_curr + (1 - momentum) * a_next
        a_next = a_next % (2*pi)
        return a_next
    

nrm = PuPy.Normalization()
nrm.set('ir', 0.0, 1.0)
nrm.set('a_curr', 0.0, 1.0)
nrm.set('a_next', 0.0, 1.0)

acd = ExperimentingHDP(data_pth, reservoir, readout, plant, policy, gamma, alpha, init_steps=5, norm=nrm)

## MAIN LOOP ##

num_episode = 0
while True:
    
    # init episode
    acd.new_episode()
    #robot.reset()
    robot.reset_random(loc_lo=-9.0, loc_hi=9.0)
    policy.action = policy.initial_action() # FIXME: Policy must be reset aswell... should this be intrinsic?
    a_curr = np.atleast_2d([policy.action])
    
    num_step = 0 # k
    while True:
        
        # Apply current action
        collided = robot.take_action(a_curr)
        
        # Observe sensors
        s_next = robot.read_sensors()
        
        # Execute ACD
        a_next = acd(s_next, num_step, num_step+1, 1)
        #a_next = policy.initial_action()
        #a_next = a_next % (2*pi)
        
        # Iterate
        num_step += 1
        #if collided or num_step >= _max_step: break
        if collided: break
        acd.a_curr = a_curr = a_next
    
    if num_step <= 3:
        print "Warning: episode ended prematurely"
    
    num_episode += 1
    #acd.set_gamma(min(gamma + 0.05, 0.5))
    
    if num_episode >= _max_episodes: break


readout.save(readout_pth)

## ANALYSIS ##

#"""
#a = rrl.Analysis(data_pth, grid=True)
a = rrl.Analysis(data_pth, min_key_length=10)

# Miscellaneous plots
rrl.overview(a, pylab.figure(1))

# Plot all trajectories
fig = pylab.figure(2)
ax = fig.add_subplot(111)
robot._plot_obstacles(with_tol=True, tol=theta)
rrl.epuck_plot_all_trajectories(a, ax)
fig.suptitle('Robot trajectories. The lighter, the earlier')

pylab.figure()
pylab.plot(a.stack_data('j_curr'), 'b', label='return')
pylab.plot(a.stack_data('reward'), 'k', label='reward')
pylab.legend(loc=0)


pylab.show(block=False)

import Oger
reward = a.stack_data('reward')
j_curr = a.stack_data('j_curr')
print "MSE", Oger.utils.mse(j_curr, reward)
print "NRMSE", Oger.utils.nrmse(j_curr, reward)
#"""


