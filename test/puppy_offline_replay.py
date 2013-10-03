import rrl
import PuPy
import numpy as np
import itertools

# Create a policy
bound_gait = {
    'amplitude' : ( 0.8, 1.0, 0.8, 1.0),
    'frequency' : (1.0, 1.0, 1.0, 1.0),
    'offset'    : ( -0.23, -0.23, -0.37, -0.37),
    'phase'     : (0.0, 0.0, 0.5, 0.5)
}

policy = rrl.puppy.policy.LRA(PuPy.Gait(bound_gait))

# Create a plant
landmarks = [i for i in itertools.product((-10.0, -3.3, 3.3, 10.0), (-10.0, -3.3, 3.3, 10.0))]
target_loc = (6.0, 4.0)
plant = rrl.puppy.plant.TargetLocationLandmarks(
    target_loc,
    landmarks,
    reward_noise    = 0.0
)

# Load the normalization
nrm = PuPy.Normalization('../data/puppy_unit.json')

# Create a reservoir
reservoir = rrl.ReservoirNode(
    output_dim      = 10,
    input_dim       = policy.action_space_dim() + plant.state_space_dim(),
    spectral_radius = 0.98,
    w               = rrl.sparse_reservoir(20),
)

reservoir.initialize()
reservoir.save('/tmp/puppy_reservoir.pic')

# Create a readout
readout = rrl.StabilizedRLS(
    with_bias       = True,
    input_dim       = reservoir.get_output_dim() + reservoir.get_input_dim(),
    output_dim      = 1,
    lambda_         = 1.0
)

# Initialize the Critic
critic = rrl.PuppyHDP(
    tumbled_reward  = 0.0,
    expfile         = '/tmp/puppy_critic.hdf5',
    reservoir       = reservoir,
    readout         = readout,
    plant           = plant,
    policy          = policy,
    gamma           = 0.5,
    alpha           = 1.0,
    init_steps      = 10,
    norm            = nrm
)

# Train the critic on offline data
rrl.puppy.offline_playback(
    '/tmp/puppy_offline_data.hdf5',
    critic,
    samples_per_action  = 150,
    ms_per_step         = 20,
    episode_start       = 0,
    episode_end         = 1000,
    min_episode_len     = 30
)

# Store the readout for later use
readout.save('/tmp/puppy_readout.pic')
