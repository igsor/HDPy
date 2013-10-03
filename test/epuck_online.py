import numpy as np
import rrl
import os
import pylab

## INITIALIZATION ##

# Robot
obstacles = [
    rrl.epuck.env.train_lower,
    rrl.epuck.env.train_middle,
    rrl.epuck.env.train_left,
    rrl.epuck.env.train_upper
]

robot = rrl.epuck.AbsoluteRobot(
    walls       = rrl.epuck.env.obstacles_box,
    obstacles   = obstacles,
    tol         = 0.0,
    speed       = 0.5,
    step_time   = 0.5,
)

# Plant and Policy
policy = rrl.epuck.policy.HeadingRandInit()
plant = rrl.epuck.plant.CollisionAvoidanceFrontal(
    theta       = 1.0,
    obs_noise   = 0.05
)

# Set up reservoir
reservoir = rrl.ReservoirNode(
    output_dim      = 50,
    input_dim       = policy.action_space_dim() + plant.state_space_dim(),
    spectral_radius = 0.95,
    input_scaling   = 1.0/3.0,
    bias_scaling    = -3.0,
    fan_in_w        = 20
)

reservoir.initialize()

# Set up readout
readout = rrl.StabilizedRLS(
    with_bias   = True,
    input_dim   = reservoir.get_output_dim() + policy.action_space_dim() + plant.state_space_dim(),
    output_dim  = 1,
    lambda_     = 1.0
)

# Custom ADHDP
class ExperimentingHDP(rrl.CollectingADHDP):
    def _next_action_hook(self, a_next):
        """Project action into the interval [0,2pi]."""
        return a_next % (2*np.pi)

# Remove old data file
if os.path.exists('/tmp/epuck_data.hdf5'):
    os.unlink('/tmp/epuck_data.hdf5')

# Create ADHDP instance
acd = ExperimentingHDP(
    # Demanded by CollectingADHDP
    expfile     = '/tmp/epuck_data.hdf5',
    # Demanded by ADHDP
    reservoir   = reservoir,
    readout     = readout,
    # Demanded by ActorCritic
    plant       = plant,
    policy      = policy,
    gamma       = 0.5,
    alpha       = 1.0,
    init_steps  = 5,
)

## SIMULATION LOOP ##

# Execute the simulation for 10 episodes, with 100 steps tops each
rrl.epuck.simulation_loop(
    acd,
    robot,
    max_step        = 100,
    max_episodes    = 10,
    max_total_iter  = -1
)

## EVALUATION ##

# Load the data file
analysis = rrl.Analysis('/tmp/epuck_data.hdf5')

# Plot the trajectories and obstacles
axis = pylab.figure().add_subplot(111)
robot._plot_obstacles(axis=axis)
rrl.epuck.plot_all_trajectories(analysis, axis)

# Show the figure
pylab.show(block=False)
