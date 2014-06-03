"""
Puppy experiments are executed within the [Webots]_ simulator. Since
this module is linked to :py:mod:`PuPy` through the class
:py:class:`ActorCritic`, this is the native approach. For the purpose of
Puppy, an adapted Actor-Critic is implemented in :py:class:`PuppyHDP`,
handling Puppy specifics. It inherits from :py:class:`ADHDP`,
hence can be used in the same fashion.

Simulation with [Webots]_ is often time consuming. Therefore, a method
is provided to collect data in the simulation and replay it later. This
is implemented through :py:class:`OfflinePuppy` and
:py:func:`puppy.offline_playback`. An example of how to approach this
is documented in :ref:`puppy_offline`.

Change log:
    02.06.2014: removed tumbling detection from PuppyHDP. Tumbling can be
                recorded by a PuPy.TumbleCollector and handled directly
                in the Plant to calculate the appropriate reward.
"""
from ..hdp import ADHDP
from ..rl import Plant
import numpy as np
import warnings
import h5py
import HDPy

SENSOR_NAMES = ['trg0', 'trg1', 'trg2', 'trg3', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'compass_x', 'compass_y', 'compass_z', 'gyro_x', 'gyro_y', 'gyro_z', 'hip0', 'hip1', 'hip2', 'hip3', 'knee0', 'knee1', 'knee2', 'knee3', 'puppyGPS_x', 'puppyGPS_y', 'puppyGPS_z', 'touch0', 'touch0', 'touch1', 'touch2', 'touch3']

class PuppyHDP(ADHDP):
    """ADHDP subtype for simulations using Puppy in webots.
    
    This class adds some code considering restarts of Puppy.
    
    """
    def __init__(self, *args, **kwargs):
        super(PuppyHDP, self).__init__(*args, **kwargs)
        self._start_new_episode = False
    
    def _signal(self, msg, **kwargs):
        """Handle messages from the supervisor. Messages are expected
        when the robot has tumbled and thus the robot has to be reset.
        """
        super(PuppyHDP, self)._signal(msg, **kwargs)        
        if msg == 'reset':
            #print "Reset received", self.num_step
            self._start_new_episode = True # remember to start a new episode at next __call__
    
    def _step(self, s_curr, epoch, a_curr, reward):
        """If reset is triggered, invoke new_episode.
        """
        a_curr_orig = a_curr.copy()
        epoch = super(PuppyHDP, self)._step(s_curr, epoch, a_curr, reward)
        if self._start_new_episode:
            self._start_new_episode = False
            self.new_episode()
            epoch['a_next'] = self.a_curr.T
            self.a_curr = a_curr_orig
        return epoch

class OfflinePuppy(PuppyHDP):
    """Collect sensor data for Puppy in webots, such that it can be
    reused later to train a critic offline.
    
    Note that in contrast to :py:class:`ADHDP`, some
    structures are not required (reservoir, plant). They will be set
    to stubs, hence don't need to be passed. 
    
    Some extra metadata is stored in the datafile, which allows
    processing of the experiment in an offline fashion through the
    function :py:func:`puppy.offline_playback`.
    
    """
    def __init__(self, *args, **kwargs):
        # look for policy's member 'action_space_dim' (policy is hidden in child or sub-child)
        action_space_dim = kwargs['policy'].action_space_dim
        if 'plant' not in kwargs:
            kwargs['plant'] = Plant(state_space_dim=0)
        plant = kwargs['plant']
        state_space_dim = plant.state_space_dim
        
        class Phony:
            """Stub for a reservoir."""
            reset_states = False
            def get_input_dim(self):
                """Return input dimension (action space dim.)"""
                return action_space_dim()+state_space_dim()
            def reset(self):
                """Reset to the initial state (no effect)"""
                pass
        
        kwargs['reservoir'] = Phony()
        kwargs['readout'] = None
        super(OfflinePuppy, self).__init__(*args, **kwargs)
    
    def __call__(self, epoch, time_start_ms, time_end_ms, step_size_ms):
        """Store the sensor measurements of an epoch in the datafile
        as well as relevant metadata. The robot detects if the
        simulation was reverted and if it has tumbled (through the
        supervisor message). Other guards are not considered, as none
        are covered by :py:class:`PuppyHDP`.
        
        """
        self.num_step += 1
        
        # Determine next action
        if self.num_step <= self._init_steps:
            # Init
            a_next = self.a_curr
            if self.verbose:
                print "(init)", a_next.T
        else:
            # Normal walking
            a_next = self._next_action_hook(self.a_curr)
            if self.verbose:
                print 't:', time_start_ms, ' a:', self.a_curr.T
        
        try:
            epoch['reward'] = np.atleast_2d(self.plant.reward(epoch))
        except NotImplementedError:
            pass
        epoch['a_curr'] = self.a_curr.T
        epoch['a_next'] = a_next.T
        self.a_curr = a_next
        
        return self.child(epoch, time_start_ms, time_end_ms, step_size_ms)
    
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

def offline_playback(pth_data, critic, samples_per_action, ms_per_step, episode_start=None, episode_end=None, min_episode_len=0, sensor_names=SENSOR_NAMES, sensor_sampling_key='trg0'):
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
        print episode_idx,'/',len(storages)
        
        data_grp = f[episode]
        N = data_grp[sensor_sampling_key].shape[0]
        assert N % samples_per_action == 0
        
        # get tumbled infos
        if 'tumble' in data_grp:
            time_tumbled = np.where(data_grp['tumble'])[0]
            if len(time_tumbled)<1:
                time_tumbled = -1
            else:  
                time_tumbled = time_tumbled[0] / samples_per_action * samples_per_action
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
            chunk = dict([(k, data_grp[k][num_iter:(num_iter+samples_per_action)]) for k in sensor_names if k in data_grp])
            
            # send tumbled message
            if num_iter == time_tumbled:
                critic.signal('tumbled')
            
            # update critic
            critic(chunk, time_start_ms, time_end_ms, time_step_ms)
        
        # send reset after episode has finished
        if episode_idx < len(storages) - 1:
            #critic.event_handler(None, dict(), ms_per_step * N, 'reset')
            critic.signal('reset')
            critic.signal('new_episode') # collectors will create new group
    
    # cleanup
    critic._pre_increment_hook = critic._pre_increment_hook_orig
    critic._next_action_hook = critic._next_action_hook_orig
    del critic._pre_increment_hook_orig
    del critic._next_action_hook_orig
    f.close()


## DEPRECATED ##

def puppy_offline_playback(*args, **kwargs):
    """Alias of offline_playback.
    
    .. deprecated:: 1.0
        Use :py:func:`offline_playback` instead
        
    """
    warnings.warn("This function is deprecated. Use 'offline_playback' instead")
    return offline_playback(*args, **kwargs)
