from controller import Robot
import PuPy
import rrl
import numpy as np


## INITIALIZATION ##

# Plant and Policy

policy = rrl.puppy.policy.

plant = rrl.puppy.plant.

# Reservoir
# FIXME: Read from file if available
reservoir = rrl.ReservoirNode(
    output_dim      = 100,
    input_dim       = policy.action_space_dim() + plant.state_space_dim(),
    reset_states    = False,
    spectral_radius = 0.7,
    w               = rrl.sparse_reservoir_gen(20),
)

reservoir.initialize()

# Readout
# FIXME: Read from file if available
# FIXME: Bring back the h5py readout storing structure
readout = rrl.StabilizedRLS(
    input_dim   = reservoir.get_output_dim() + reservoir.get_input_dim(),
    output_dim  = 1,
    with_bias   = True,
    lambda_     = 1.0
)

# Acting schema
class OnlinePuppy(rrl.PuppyHDP):
    def _next_action_hook(self, a_next):
        """
        
        .. todo::
            eps-greedy, documentation
        
        """
        a_next[a_next < 0.2] = 0.2
        a_next[a_next > 1.0] = 1.0
        return a_next

# actor
actor = OnlinePuppy(
    # 
    tumbled_reward  = 0.0,
    # 
    expfile         = '/tmp/puppy_online.hdf5',
    # 
    reservoir       = reservoir,
    readout         = readout,
    # 
    plant           = plant,
    policy          = policy,
    gamma           = 0.5,
    alpha           = 1.0,
    init_steps      = 10,
    norm            = nrm
)

# robot
r = PuPy.robotBuilder(
    Robot,
    actor,
    sampling_period_ms  = 20,
    ctrl_period_ms      = 3000,
    event_handlers      = actor.event_handler,
)

## SIMULATION LOOP ##

# run the simulation
r.run()

# teardown
readout.save('/tmp/puppy_readout.pic')
