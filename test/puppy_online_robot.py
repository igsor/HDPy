
from controller import Robot
import PuPy
import rrl
import os.path
import numpy as np
from config import *
from math import exp
import h5py
import sys
sys.path.append('/home/matthias/studium/master/work/hdp_variations') # makes algorithms module available
import algorithms

### WARNING: DUE TO REVERTING, NUM_EPISODE CANNOT SIMPLY BE USED ###
# The episode number from the datafile must be used, see rrl.CollectingADHDP

#reservoir_pth   = '/home/matthias/Desktop/puppy/online/online_bs/puppy_hdp_reservoir.pic'
#readout_pth     = '/home/matthias/Desktop/puppy/online/online_bs/puppy_hdp_readout.pic'
#online_data_pth = '/home/matthias/Desktop/puppy/online/online_bs/puppy_hdp.hdf5'
reservoir_pth   = '/tmp/puppy_hdp_reservoir.pic'
readout_pth     = '/tmp/puppy_hdp_readout.pic'
online_data_pth = '/tmp/puppy_hdp.hdf5'

def gamma(num_episode, num_step):
    return min(0.5, 0.001 * num_episode)

if reservoir_pth is not None and os.path.exists(reservoir_pth):
    print "Loading reservoir"
    import pickle
    f = open(reservoir_pth, 'r')
    reservoir = pickle.load(f)
    reservoir.reset()
    f.close()
else:
    reservoir = rrl.SparseReservoirNode(
        output_dim=100,
        input_dim=policy.action_space_dim() + plant.state_space_dim(),
        reset_states=False,
        spectral_radius=0.7,
        input_scaling=1.0,
        bias_scaling=0.0,
        fan_in_i=100,
        fan_in_w=20,
    )
    
    reservoir.initialize()
    reservoir.save(reservoir_pth + '.latest')

readout = rrl.StabilizedRLS(
    input_dim=reservoir.get_output_dim() + reservoir.get_input_dim(),
    output_dim=1,
    with_bias=True,
    lambda_=1.0
)

if readout_pth is None or len(readout_pth) == 0:
    raise Exception('Readout must not be empty')

readout_pth += '_data.hdf5'
if os.path.exists(readout_pth):
    rf = h5py.File(readout_pth,'a')
    readout._psiInv = rf['psiInv'][:]
    readout.beta = rf['beta'][:]
else:
    print "Creating readout data file"
    N = reservoir.get_output_dim() + 1
    rf = h5py.File(readout_pth,'w')
    rf.create_dataset('psiInv', data=readout._psiInv)
    rf.create_dataset('beta', data=readout.beta)

class OnlinePuppy(rrl.PuppyHDP):
    def _pre_increment_hook(self, *args, **kwargs):
        rf['psiInv'][:] = readout._psiInv
        rf['beta'][:] = readout.beta
        super(OnlinePuppy, self)._pre_increment_hook(*args, **kwargs)
    
    def _next_action_hook(self, a_next):
        a_next[a_next < 0.2] = 0.2
        a_next[a_next > 1.0] = 1.0
        #eps = 0.2
        if np.random.uniform(0.0, 1.0) < eps:
            a_cand = np.zeros(a_next.shape)
            while (a_cand < 0.2).any() or (a_cand > 2.0).any() or a_cand.ptp() > 0.5:
                a_cand = a_next + np.random.normal(0.0, 0.25, size=a_next.shape)
            
            a_next = a_cand
        
        return a_next

# actor
actor = OnlinePuppy(
    tumbled_reward=0.0,
    expfile=online_data_pth,
    reservoir=reservoir,
    readout=readout,
    plant=plant,
    policy=policy,
    gamma=gamma,
    alpha=alpha,
    init_steps=25,
    norm=nrm
)

# robot
r = PuPy.robotBuilder(
    Robot,
    actor,
    sampling_period_ms=20,
    ctrl_period_ms=3000,
    event_handlers=actor.event_handler,
)

# run
r.run()
