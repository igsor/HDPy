"""
HDP code for Puppy experiments

"""
from rl import CollectingADHDP
import numpy as np

class PuppyHDP(CollectingADHDP):
    """ADHDP subtype for simulations using Puppy in webots.
    
    This class adds some code considering restarts of Puppy. It adds
    an optional argument ``tumbled_reward``. The reward will be forced
    to this value after the supervisor detected tumbling. If
    :py:keyword:`None` (the default) is used, the reward remains
    unchanged.
    
    """
    def __init__(self, *args, **kwargs):
        self._tumbled_reward = kwargs.pop('tumbled_reward', None)
        self.has_tumbled = False
        self.supervisor_tumbled_notice = 0
        super(PuppyHDP, self).__init__(*args, **kwargs)
    
    def event_handler(self, robot, epoch, current_time, msg):
        """Handle messages from the supervisor. Messages are expected
        when the robot has tumbled and thus the robot has to be reset.
        """
        # msg is 'reset', 'out_of_arena', 'tumbled_grace_start' or 'tumbled'
        # for msg==reset, the robot is reset immediately
        # msg==tumbled_grace_start marks the start of the grace period of the tumbled robot
        if msg == 'tumbled_grace_start':
            self.supervisor_tumbled_notice = 1
        
        if msg == 'reset':
            self.policy.reset()
            self.new_episode()
    
    def new_episode(self):
        """After restarting, reset the tumbled values and start the
        new episode.
        """
        super(PuppyHDP, self).new_episode()
        self.has_tumbled = False
        self.supervisor_tumbled_notice = 0
    
    def _step(self, s_curr, s_next, a_curr, reward):
        """Ensure the tumbled reward and initiate behaviour between
        restarts. The step of the parent is then invoked.
        """
        if self.has_tumbled:
            return np.zeros(shape=a_curr.shape)
        
        if self.supervisor_tumbled_notice > 0:
            if self.supervisor_tumbled_notice > 1:
                if self._tumbled_reward is not None:
                    reward = self._tumbled_reward
                self.has_tumbled = True
            self.supervisor_tumbled_notice += 1
        
        a_next = super(PuppyHDP, self)._step(s_curr, s_next, a_curr, reward)
        print reward, a_curr.T, a_next.T
        return a_next
    
    def init_episode(self, epoch, time_start_ms, time_end_ms, step_size_ms):
        """Initial behaviour (after reset)
        
        .. note::
            Assuming identical initial trajectories, the initial state
            is the same - and thus doesn't matter.
            Non-identical initial trajectories will result in
            non-identical behaviour, therefore the initial state should
            be different (initial state w.r.t. start of learning).
            Due to this, the critic is already updated in the initial
            trajectory.
        """
        if self.num_step > 2:
            in_state = self.plant.state_input(self.s_curr)
            a_curr_nrm = self.normalizer.normalize_value('a_curr', self.a_curr)
            i_curr = np.vstack((in_state, a_curr_nrm)).T
            x_curr = self.reservoir(i_curr, simulate=False)
            self._pre_increment_hook(
                epoch,
                x_curr=x_curr,
                i_curr=i_curr,
                a_next=self.a_curr.T,
                a_curr=self.a_curr.T,
            )
        
        self.s_curr = epoch
        return self.policy.get_iterator(time_start_ms, time_end_ms, step_size_ms)


