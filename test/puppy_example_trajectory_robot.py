from controller import Robot
import PuPy
import HDPy
import numpy as np
import h5py

# Initialize a policy
bound_gait = {
    'amplitude' : ( 0.8, 1.0, 0.8, 1.0),
    'frequency' : (1.0, 1.0, 1.0, 1.0),
    'offset'    : ( -0.23, -0.23, -0.37, -0.37),
    'phase'     : (0.0, 0.0, 0.5, 0.5)
}

policy = HDPy.puppy.policy.LRA(PuPy.Gait(bound_gait, 'bounding'))

# OfflineCollector which follows a predefined sequence of actions
# after the initial behaviour (policy with default params for 25 steps).
class TrajectoryFollower(HDPy.puppy.OfflineCollector):
    def __init__(self, trajectory, *args, **kwargs):
        super(TrajectoryFollower, self).__init__(*args, **kwargs)
        self.trajectory = trajectory
        self._traj_idx = 0
    
    def _next_action_hook(self, a_next):
        if self._traj_idx >= self.trajectory.shape[0]:
            # If all actions have been executed, signal the supervisor
            # to revert the simulation
            self.robot.send_msg('revert_on_demand')
            return self.a_curr
            
        # If there's a next action, execute it
        a_next = np.atleast_2d(self.trajectory[self._traj_idx]).T
        self._traj_idx += 1
        
        return a_next

# Load the sequence file
f = h5py.File('/tmp/example_sequence.hdf5','a')
# Get the index of the trajectory to be executed
idx = f['idx'][()]
grp_name = 'traj_%03i' % idx
if grp_name in f:
    # Not yet finished, increment the index such that the next
    # trajectory is executed in the next experiment.
    trajectory = f[grp_name][:]
    do_quit = False
    f['idx'][()] += 1
else:
    # Simulation is finished, execute any trajectory and prepare for
    # termination
    while grp_name not in f and idx >= 0:
        idx -= 1
        grp_name = 'traj_%03i' % (idx)
    
    if idx < 0:
        raise Exception('Could not find last trajectory')
    
    trajectory = f[grp_name][:]
    do_quit = True

f.close()

# Initialize the collector
collector = PuPy.RobotCollector(
    child   = policy,
    expfile = '/tmp/example_data.hdf5'
)

# Initialize the actor
actor = TrajectoryFollower(
    trajectory  = trajectory,
    policy      = collector,
    init_steps  = 10,
)

# Initialize the robot, bind it to webots
r = PuPy.robotBuilder(
    Robot,
    actor,
    sampling_period_ms  = 20,
    ctrl_period_ms      = 3000,
)

# Register robot in actor for signalling
actor.robot = r

if do_quit:
    # Quit the simulation when all trajectories are handled
    r.send_msg('quit_on_demand')

# Run the simulation
r.run()
