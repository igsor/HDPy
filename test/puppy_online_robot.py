from controller import Robot
import PuPy
import HDPy
import numpy as np
import os
import itertools
import pickle


## INITIALIZATION ##

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
    reward_noise    = 0.0
)
# Load the normalization
nrm = PuPy.Normalization(os.path.split(HDPy.__file__)[0]+'/../data/puppy_unit.json')

# Reservoir
if os.path.exists('/tmp/puppy_reservoir.pic'):
    reservoir = pickle.load(open('/tmp/puppy_reservoir.pic','r'))
else:
    reservoir = HDPy.ReservoirNode(
        output_dim      = 100,
        input_dim       = policy.action_space_dim() + plant.state_space_dim(),
        reset_states    = False,
        spectral_radius = 0.7,
        w               = HDPy.sparse_reservoir(20),
    )
    reservoir.initialize()
    reservoir.save('/tmp/puppy_reservoir.pic')

# Readout
if os.path.exists('/tmp/puppy_readout.pic'):
    readout = pickle.load(open('/tmp/puppy_readout.pic','r'))
else:
    readout = HDPy.StabilizedRLS(
        input_dim   = reservoir.get_output_dim() + reservoir.get_input_dim(),
        output_dim  = 1,
        with_bias   = True,
        lambda_     = 1.0
    )

# Acting schema
class OnlinePuppy(HDPy.PuppyHDP):
    def _next_action_hook(self, a_next):
        """Choose the action in an eps-greedy
        fashion, meaning that a random action
        is preferred over the suggested one with
        probability eps.
        """
        if np.random.rand() < 0.2:
            a_next = np.random.uniform(low=0.2, high=1.0, size=a_next.shape)
        # clip the action to a bounded range
        a_next[a_next < 0.2] = 0.2
        a_next[a_next > 1.0] = 1.0
        return a_next

# Initialize the collector
collector = PuPy.RobotCollector(
    child   = policy,
    expfile = '/tmp/puppy_online.hdf5'
)

# actor
actor = OnlinePuppy(
    # HDPy.puppy.PuppyHDP
    tumbled_reward  = 0.0,
    # HDPy.ADHDP
    reservoir       = reservoir,
    readout         = readout,
    # HDPy.ActorCritic
    plant           = plant,
    policy          = collector,
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
#    event_handlers      = actor.event_handler
)

## SIMULATION LOOP ##

# run the simulation
r.run()

# teardown
readout.save('/tmp/puppy_readout.pic')
