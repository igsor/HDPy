"""
HDP code for Puppy experiments

"""
from rl import CollectingADHDP, Plant
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
            #print "Tumbling received", self.num_step
            self.supervisor_tumbled_notice = 1
        
        if msg == 'reset':
            #print "Reset received", self.num_step
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
                
                reward /= (1.0 - self.gamma(self.num_episode, self.num_step))
                # geometric series to incorporate future rewards
                # note that with this, its err = r/(1-gamma) - J * (1-gamma)
                # but should be err = r/(1-gamma) - J
                # thus, there's an difference of J*gamma
                # could, for example solve this by temporarily set gamma = 0.0
                self.has_tumbled = True
                old_gamma = self.gamma
                self.set_gamma(0.0)
            self.supervisor_tumbled_notice += 1
        
        a_next = super(PuppyHDP, self)._step(s_curr, s_next, a_curr, reward)
        
        if self.supervisor_tumbled_notice > 2:
            self.gamma = old_gamma
        
        #print self.num_step, reward, a_curr.T, a_next.T, s_next['puppyGPS_x'][-1]
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

class OfflineCollector(CollectingADHDP):
    def __init__(self, *args, **kwargs):
        class Phony:
            reset_states = False
            def get_input_dim(self):
                return kwargs['policy'].action_space_dim()
            def reset(self):
                pass
        
        kwargs['plant'] = Plant(state_space_dim=0)
        kwargs['reservoir'] = Phony()
        kwargs['readout'] = None
        self.supervisor_tumbled_notice = 0
        super(OfflineCollector, self).__init__(*args, **kwargs)
    
    def new_episode(self):
        """After restarting, reset the tumbled values and start the
        new episode.
        """
        super(OfflineCollector, self).new_episode()
        self.supervisor_tumbled_notice = 0
    
    def __call__(self, epoch, time_start_ms, time_end_ms, step_size_ms):
        #print "(call)", time_start_ms, self.a_curr.T, ('puppyGPS_x' in epoch and epoch['puppyGPS_x'][-1] or 'NOX')
        if len(epoch) == 0:
            # the very first initial epoch of the first episode
            # this case occurs when the simulation starts or after it is reverted
            self.num_step += 1
            #self._pre_increment_hook(dict(), empty_initial_step=np.array([1]))
            self._pre_increment_hook(dict(), init_step=np.array([self.num_step]))
            return self.policy.get_iterator(time_start_ms, time_end_ms, step_size_ms)
        
        # Determine next action
        if self.num_step <= self._init_steps:
            # Init
            a_next = self.a_curr
        elif self.supervisor_tumbled_notice > 2:
            # Tumbled, prepare for reset
            a_next = np.zeros(shape=self.a_curr.shape)
            self.supervisor_tumbled_notice += 1
        elif self.supervisor_tumbled_notice > 0:
            # Tumbled, still walking
            a_next = self._next_action_hook(self.a_curr)
            self.supervisor_tumbled_notice += 1
        else:
            # Normal walking
            a_next = self._next_action_hook(self.a_curr)
        
        if self.num_step <= self._init_steps:
            print "(init)", a_next.T
        elif self.supervisor_tumbled_notice > 2:
            print time_start_ms, self.a_curr.T, self.num_step
        else:
            print time_start_ms, self.a_curr.T, epoch['puppyGPS_x'][-1]
        
        self._pre_increment_hook(
            epoch,
            a_curr=self.a_curr.T,
            a_next=a_next.T
        )
        
        self.a_curr = a_next
        self.num_step += 1
        
        self.policy.update(a_next)
        #print "(call-end)", self.num_step, a_next.T
        return self.policy.get_iterator(time_start_ms, time_end_ms, step_size_ms)
    
    def event_handler(self, robot, epoch, current_time, msg):
        """Handle messages from the supervisor. Messages are expected
        when the robot has tumbled and thus the robot has to be reset.
        """
        # msg is 'reset', 'out_of_arena', 'tumbled_grace_start' or 'tumbled'
        # for msg==reset, the robot is reset immediately
        # msg==tumbled_grace_start marks the start of the grace period of the tumbled robot
        if msg == 'tumbled_grace_start':
            #print "Tumbling received", self.num_step
            self.supervisor_tumbled_notice = 1
            self._pre_increment_hook(dict(), tumbled=np.array([self.num_step]))
        
        if msg == 'reset':
            #print "Reset received", self.num_step
            self.policy.reset()
            self.new_episode()
    
    def _next_action_hook(self, a_next):
        a_next = np.zeros(self.a_curr.shape)
        while (a_next < 0.2).any() or (a_next > 2.0).any() or ((a_next > 1.0).any() and a_next.ptp() > 0.4): # Prohibit too small or large amplitudes
            #a_next = self.a_curr + np.random.uniform(-0.2, 0.2, size=self.a_curr.shape)
            a_next = self.a_curr + np.random.normal(0.0, 0.15, size=self.a_curr.shape)
        
        return a_next

