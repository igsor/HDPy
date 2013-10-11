from controller import Robot
import HDPy
import PuPy
import numpy as np

# Policy setup
bound_gait = {
    'amplitude' : ( 0.8, 1.0, 0.8, 1.0),
    'frequency' : (1.0, 1.0, 1.0, 1.0),
    'offset'    : ( -0.23, -0.23, -0.37, -0.37),
    'phase'     : (0.0, 0.0, 0.5, 0.5)
}

policy = HDPy.puppy.policy.LRA(PuPy.Gait(bound_gait))

def random_initial_action():
    """Select a random action initially instead of a fixed one,
    specified by the gait definition.
    """
    N = policy.action_space_dim()
    action = np.atleast_2d([-1.0] * N)
    while (action < 0.4).any() or (action > 2.0).any() or action.ptp() > 0.5:
        action = np.random.normal(0.9, 0.3, size=action.shape)
    return action.T

policy.initial_action = random_initial_action

# Offline data collector setup
class OfflinePuppy(HDPy.puppy.OfflineCollector):
    def _next_action_hook(self, a_next):
        """Define the schema according to which actions will be selected.
        Hence, this function defines the action and state space sampling
        schema. Note that the influence on training is intense.
        
        """
        a_next = np.zeros(self.a_curr.shape)
        # Prohibit too small or large amplitudes
        while (a_next < 0.2).any() or (a_next > 2.0).any() or ((a_next > 1.0).any() and a_next.ptp() > 0.4):
            a_next = self.a_curr + np.random.normal(0.0, 0.15, size=self.a_curr.shape)
        
        return a_next

# actor instantiation
actor = OfflinePuppy(
    expfile     = '/tmp/puppy_offline_data.hdf5',
    policy      = policy,
    init_steps  = 10,
)

# robot instantiation
r = PuPy.robotBuilder(
    Robot,
    actor,
    sampling_period_ms  = 20,
    ctrl_period_ms      = 3000,
    event_handlers      = actor.event_handler,
)

# invoke the main loop, starts the simulation
r.run()
