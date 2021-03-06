import HDPy
import pickle
import os
import sys
import PuPy
import itertools

# Load reservoir
f = open('/tmp/puppy_reservoir.pic', 'r')
reservoir = pickle.load(f)
reservoir.reset()
f.close()

# Load readout
f = open('/tmp/puppy_readout.pic', 'r')
readout = pickle.load(f)
f.close()

# Critic is evaluated, thus don't train it anymore
readout.stop_training()

# Create a policy
bound_gait = {
    'amplitude' : ( 0.8, 1.0, 0.8, 1.0),
    'frequency' : (1.0, 1.0, 1.0, 1.0),
    'offset'    : ( -0.23, -0.23, -0.37, -0.37),
    'phase'     : (0.0, 0.0, 0.5, 0.5)
}

policy = HDPy.puppy.policy.LRA(PuPy.Gait(bound_gait))

# Create a plant
landmarks = [i for i in itertools.product((-10.0, -3.3, 3.3, 10.0), (-10.0, -3.3, 3.3, 10.0))]
target_loc = (6.0, 4.0)
plant = HDPy.puppy.plant.TargetLocationLandmarks(
    target_loc,
    landmarks,
    reward_noise = 0.0
)

# Load the normalization
nrm = PuPy.Normalization('../data/puppy_unit.json')

# Initialize the collector
collector = PuPy.RobotCollector(
    child   = policy,
    expfile = '/tmp/example_eval.hdf5'
)

# Create HDP instance
actor = HDPy.PuppyHDP(
    tumbled_reward  =0.0,
    reservoir       = reservoir,
    readout         = readout,
    plant           = plant,
    policy          = collector,
    gamma           = 0.0,
    alpha           = 1.0,
    init_steps      = 10,
    norm            = nrm
)

HDPy.puppy.offline_playback(
    '/tmp/example_data.hdf5',
    actor,
    150,
    20
)

