"""
Puppy experiments are executed within the [Webots]_ simulator. This
process is supported by :py:mod:`PuPy`. For this purpose, an adapted
Actor-Critic is implemented in :py:class:`PuppyHDP`, handling Puppy
specifics. It inherits from :py:class:`CollectingADHDP`, hance can be
used in the same fashion.

Simulation with [Webots]_ is often time consuming. Therefore, a method
is provided to collect data in the simulation and replay it later. This
is implemented through :py:class:`OfflineCollector` and
:py:func:`puppy_offline_playback`. An example of how to approach this
is documented in :ref:`puppy_offline`.

"""
from ..hdp import CollectingADHDP
from ..rl import Plant
import numpy as np
import warnings
import h5py

SENSOR_NAMES = ['trg0', 'trg1', 'trg2', 'trg3', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'compass_x', 'compass_y', 'compass_z', 'gyro_x', 'gyro_y', 'gyro_z', 'hip0', 'hip1', 'hip2', 'hip3', 'knee0', 'knee1', 'knee2', 'knee3', 'puppyGPS_x', 'puppyGPS_y', 'puppyGPS_z', 'touch0', 'touch0', 'touch1', 'touch2', 'touch3']

class PuppyHDP(CollectingADHDP):
    """ADHDP subtype for simulations using Puppy in webots.
    
    This class adds some code considering restarts of Puppy. It adds
    an optional argument ``tumbled_reward``. The reward will be forced
    to this value after the supervisor detected tumbling. If
    :py:const:`None` (the default) is used, the reward remains
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
                    reward = np.atleast_2d([self._tumbled_reward])
                
                #reward /= (1.0 - self.gamma(self.num_episode, self.num_step))
                # geometric series to incorporate future rewards
                # note that with this, its err = r/(1-gamma) - J * (1-gamma)
                # but should be err = r/(1-gamma) - J
                # thus, there's an difference of J*gamma
                # is solved this by temporarily set gamma = 0.0
                self.has_tumbled = True
                #old_gamma = self.gamma
                #self.set_gamma(0.0)
            self.supervisor_tumbled_notice += 1
        
        reward += np.random.normal(scale=0.001)
        a_next = super(PuppyHDP, self)._step(s_curr, s_next, a_curr, reward)
        
        #if self.supervisor_tumbled_notice > 2:
        #    self.gamma = old_gamma
        
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
            x_curr = np.hstack((x_curr, i_curr)) # FIXME: Input/Output ESN Model
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
    """Collect sensor data for Puppy in webots, such that it can be
    reused later to train a critic offline.
    
    Note that in contrast to :py:class:`CollectingADHDP`, some
    structures are not required (reservoir, plant). They will be set
    to stubs, hence don't need to be passed. 
    
    Some extra metadata is stored in the datafile, which allows
    processing of the experiment in an offline fashion through the
    function :py:func:`puppy_offline_playback`.
    
    """
    def __init__(self, *args, **kwargs):
        class Phony:
            """Stub for a reservoir."""
            reset_states = False
            def get_input_dim(self):
                """Return input dimension (action space dim.)"""
                return kwargs['policy'].action_space_dim()
            def reset(self):
                """Reset to the initial state (no effect)"""
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
        """Store the sensor measurements of an epoch in the datafile
        as well as relevant metadata. The robot detects if the
        simulation was reverted and if it has tumbled (through the
        supervisor message). Other guards are not considered, as none
        are covered by :py:class:`PuppyHDP`.
        
        """
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
        """Defines the action sampling policy of the offline data
        gathering. Note that this policy is very relevant to later
        experiments, hence this methods should be overloaded (although
        a default policy is provided).
        """
        warnings.warn('Default sampling policy is used.')
        a_next = np.zeros(self.a_curr.shape)
        # Prohibit too small or large amplitudes
        while (a_next < 0.2).any() or (a_next > 2.0).any() or ((a_next > 1.0).any() and a_next.ptp() > 0.4):
            a_next = self.a_curr + np.random.normal(0.0, 0.15, size=self.a_curr.shape)
        
        return a_next

def puppy_offline_playback(pth_data, critic, samples_per_action, ms_per_step, episode_start=None, episode_end=None, min_episode_len=0):
    """Simulate an experiment run for the critic by using offline data.
    The data has to be collected in webots, using the respective
    robot and supervisor. Note that the behaviour of the simulation
    should match what's expected by the critic. The critic is fed the
    sensor data, in order. Of course, it can't react to it since
    the next action is predefined.
    
    Additional to the sensor fields, the 'tumbling' dataset is expected
    which indicates, if and when the robot has tumbled. It is used such
    that the respective signals can be sent to the critic.
    
    The critic won't store any sensory data again.
    
    ``pth_data``
        Path to the datafile with the sensory information (HDF5).
    
    ``critic``
        PuppyHDP instance.
    
    ``samples_per_action``
        Number of samples per control step. Must correspond to the data.
    
    ``ms_per_step``
        Sensor sampling period.
    
    ``episode_start``
        Defines a lower limit on the episode number. Passed as int,
        is with respect to the episode index, not its identifier.
    
    ``episode_stop``
        Defines an upper limit on the episode number. Passed as int,
        is with respect to the episode index, not its identifier.
    
    ``min_episode_len``
        Only pick episodes longer than this threshold.
    
    """
    # Open data file, get valid experiments
    f = h5py.File(pth_data,'r')
    storages = map(str, sorted(map(int, f.keys())))
    storages = filter(lambda s: len(f[s]) > 0, storages)
    if min_episode_len > 0:
        storages = filter(lambda s: f[s]['a_curr'].shape[0] > min_episode_len, storages)
    
    if episode_end is not None:
        storages = storages[:episode_end]
    
    if episode_start is not None:
        storages = storages[episode_start:]
    
    assert len(storages) > 0
    
    # Prepare critic; redirect hooks to avoid storing epoch data twice
    # and feed the actions
    next_action = None
    episode = None
    critic._pre_increment_hook_orig = critic._pre_increment_hook
    critic._next_action_hook_orig = critic._next_action_hook
    
    def pre_increment_hook(epoch, **kwargs):
        kwargs['offline_episode'] = np.array([episode])
        critic._pre_increment_hook_orig(dict(), **kwargs)
    def next_action_hook(a_next):
        #print "(next)", a_next.T, next_action.T
        return next_action
    
    critic._next_action_hook = next_action_hook
    critic._pre_increment_hook = pre_increment_hook
    
    # Main loop, feed data to the critic
    time_step_ms = ms_per_step * samples_per_action
    time_start_ms = 0
    for episode_idx, episode in enumerate(storages):
        
        data_grp = f[episode]
        N = data_grp['trg0'].shape[0]
        assert N % samples_per_action == 0
        
        # get tumbled infos
        if 'tumbled' in data_grp:
            time_tumbled = data_grp['tumbled'][0] * samples_per_action
        else:
            time_tumbled = -1
        
        # initial, empty call
        if 'init_step' in data_grp:
            print "Simulation was started/reverted"
            time_start_ms = 0
            critic(dict(), time_start_ms, time_start_ms + samples_per_action, ms_per_step)
            time_tumbled -= samples_per_action
        
        # initial action
        critic.a_curr = np.atleast_2d(data_grp['a_curr'][0]).T
        
        # loop through data, incrementally feed the critic
        for num_iter in np.arange(0, N, samples_per_action):
            # next action
            next_action = np.atleast_2d(data_grp['a_next'][num_iter/samples_per_action]).T
            
            # get data
            time_start_ms += time_step_ms
            time_end_ms = time_start_ms + time_step_ms
            chunk = dict([(k, data_grp[k][num_iter:(num_iter+samples_per_action)]) for k in SENSOR_NAMES])
            
            # send tumbled message
            if num_iter == time_tumbled:
                critic.event_handler(None, dict(), time_tumbled, 'tumbled_grace_start')
            
            # update critic
            critic(chunk, time_start_ms, time_end_ms, time_step_ms)
        
        # send reset after episode has finished
        if episode_idx < len(storages) - 1:
            critic.event_handler(None, dict(), ms_per_step * N, 'reset')
    
    # cleanup
    critic._pre_increment_hook = critic._pre_increment_hook_orig
    critic._next_action_hook = critic._next_action_hook_orig
    del critic._pre_increment_hook_orig
    del critic._next_action_hook_orig

