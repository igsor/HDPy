from controller import Robot
import PuPy
import rrl
import numpy as np
import pickle

plant=, policy=, gamma=, nrm=

f = open(reservoir_pth, 'r')
reservoir = pickle.load(f)
reservoir.reset()
f.close()

f = open(readout_pth, 'r')
readout = pickle.load(f)
f.close()
readout.stop_training()

# actor
actor = builder(
    PuppyHDP,
    tumbled_reward=0.0,
    expfile='/tmp/example_actor.hdf5',
    reservoir=reservoir,
    readout=readout,
    plant=plant,
    policy=policy,
    gamma=gamma,
    alpha=1.0,
    init_steps=25,
    norm=nrm,
)

# robot
r = PuPy.robotBuilder(
    Robot,
    actor,
    sampling_period_ms=20,
    ctrl_period_ms=3000,
    event_handlers=actor.event_handler
)

# run
r.run()
