"""Heuristic Dynamic Programming code.

In this file, the HDP and ADHDP algorithms are implemented. They build
on top of the Actor-Critic class in the file *rl.py*. For the basic
ADHDP algorithm, a collecting version exists, which stores the data
in a HDF5 file (through PuPy.PuppyCollector)

"""
import numpy as np
import PuPy
from rl import ActorCritic


class ADHDP(ActorCritic):
    """Action dependent Heuristic Dynamic Programming structure and
    basic algorithm implementation.
    
    In the :py:meth:`_step` method, this class provides the
    implementation of a baseline algorithm. By default, the behaviour
    is online, i.e. the critic is trained and the actor in effect.
    Note that the actor can be modified through the
    :py:meth:`_next_action_hook` routine.
    
    ``reservoir``
        Critic reservoir. Should have been initialized.
    
    ``readout``
        Reservoir readout. Usually an online linear regression (RLS)
        instance, like :py:class:`StabilizedRLS`.
    
    """
    def __init__(self, reservoir, readout, *args, **kwargs):
        self.reservoir = reservoir
        self.readout = readout
        super(ADHDP, self).__init__(*args, **kwargs)
        
        # Check assumptions
        assert self.reservoir.reset_states == False
        assert self.reservoir.get_input_dim() == self.policy.action_space_dim() + self.plant.state_space_dim()
        assert self.reservoir.get_input_dim() >= self.policy.initial_action().shape[0]
    
    def new_episode(self):
        """Start a new episode of the same experiment. This method can
        also be used to initialize the *ActorCritic*, for example when
        it is loaded from a file.
        """
        self.reservoir.reset()
        self.plant.reset()
        super(ADHDP, self).new_episode()
    
    def _critic_eval(self, state, action, simulate, action_name='a_curr'):
        """Evaluate the critic at ``state`` and ``action``."""
        in_state = self.plant.state_input(state)
        action_nrm = self.normalizer.normalize_value(action_name, action)
        r_input = np.vstack((in_state, action_nrm)).T
        #r_input += np.random.normal(scale=0.001, size=r_input.shape)
        r_state = self.reservoir(r_input, simulate=simulate)
        #o_state = r_state # TODO: Direct ESN Model
        o_state = np.hstack((r_state, r_input)) # TODO: Input/Output ESN Model
        j_curr = self.readout(o_state)
        return r_input, o_state, j_curr
    
    def _critic_deriv_io_model(self, r_state):
        """Return the critic's derivative at ``r_state``."""
        direct_input_size = self.plant.state_space_dim()+self.policy.action_space_dim() # Input/Output ESN Model
        r_state = r_state[:, :-direct_input_size] # this is because _critic_eval appends the input to the state
        dtanh = (np.ones(r_state.shape) - r_state**2).T # Nx1
        dstate = dtanh * self.reservoir.w_in[:, -self._motor_action_dim:].toarray() # Nx1 .* NxA => NxA
        deriv = self.readout.beta[1:-direct_input_size].T.dot(dstate) # Input/Output ESN Model
        deriv += self.readout.beta[-self._motor_action_dim:].T # Input/Output ESN Model
        deriv = deriv.T # AxL
        scale = self.normalizer.get('a_curr')[1]
        deriv *= scale # Derivative denormalization
        return deriv
    
    def _critic_deriv_direct_model(self, r_state):
        """Return the critic's derivative at ``r_state``."""
        dtanh = (np.ones(r_state.shape) - r_state**2).T # Nx1
        dstate = dtanh * self.reservoir.w_in[:, -self._motor_action_dim:].toarray() # Nx1 .* NxA => NxA
        deriv = self.readout.beta[1:].T.dot(dstate) #  LxA # Direct ESN Model
        deriv = deriv.T # AxL
        scale = self.normalizer.get('a_curr')[1]
        deriv *= scale # Derivative denormalization
        return deriv
    
    def _critic_deriv(self, r_state):
        """Return the critic's derivative at ``r_state``."""
        return self._critic_deriv_io_model(r_state)
    
    def _step(self, s_curr, s_next, a_curr, reward):
        """Execute one step of the actor and return the next action.
        
        ``s_next``
            Latest observed state. :py:keyword:`dict`, same as ``s_next``
            of the :py:meth:`__call__`.
        
        ``s_curr``
            Previous observed state. :py:keyword:`dict`, same as ``s_next``
            of the :py:meth:`__call__`.
        
        ``reward``
            Reward of ``s_next``
        
        """
        # ESN-critic, first instance: in(k) => J(k)
        i_curr, x_curr, j_curr = self._critic_eval(s_curr, a_curr, simulate=False, action_name='a_curr')
        
        # Next action
        deriv = self._critic_deriv(x_curr)
        
        # gradient training of action (acc. to eq. 10)
        #a_next = a_curr + self.alpha(self.num_episode, self.num_step) * deriv
        a_next = a_curr + self.alpha(self.num_episode, self.num_step) * deriv
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, x_next, j_next = self._critic_eval(s_next, a_next, simulate=True, action_name='a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        self.readout.train(x_curr, err=err)
        
        # increment hook
        self._pre_increment_hook(
            s_next,
            reward=np.atleast_2d([reward]).T,
            deriv=deriv.T,
            err=err.T,
            readout=self.readout.beta.T,
            gamma=np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T,
            i_curr=i_curr,
            x_curr=x_curr,
            j_curr=j_curr,
            a_curr=a_curr.T,
            i_next=i_next,
            x_next=x_next,
            j_next=j_next,
            a_next=a_next.T
            )
        
        # increment
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
        if self.num_step > 1:
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

class CollectingADHDP(ADHDP):
    """Actor-Critic design with data collector.
    
    A :py:class:`PuPy.PuppyCollector` instance is created for recording
    sensor data and actor-critic internals together. The file is stored
    at ``expfile``.
    """
    def __init__(self, expfile, *args, **kwargs):
        self.expfile = expfile
        self.collector = None
        
        self.headers = None
        if 'additional_headers' in kwargs:
            self.headers = kwargs.pop('additional_headers')
        
        super(CollectingADHDP, self).__init__(*args, **kwargs)
    
    def _pre_increment_hook(self, epoch, **kwargs):
        """Add the *ADHDP* internals to the epoch and use the
        *collector* to save all the data.
        """
        ep_write = epoch.copy()
        for key in kwargs:
            ep_write[key] = kwargs[key]
        
        self.collector(ep_write, 0, 1, 1)
    
    def save(self, pth):
        """Save the instance without the *collector*.
        
        .. note::
            When an instance is reloaded via :py:meth:`ADHDP.load`
            a new group will be created in *expfile*.
        
        """
        # Shift collector to local
        collector = self.collector
        self.collector = None
        super(CollectingADHDP, self).save(pth)
        # Shift collector to class
        self.collector = collector

    def new_episode(self):
        """Do everything the parent does and additionally reinitialize
        the collector. The reservoir weights are stored as header
        information.
        """
        super(CollectingADHDP, self).new_episode()
        
        # init/reset collector
        if self.collector is not None:
            del self.collector
        
        self.collector = PuPy.PuppyCollector(
            actor=None,
            expfile=self.expfile,
            headers=self.headers
            #headers={
            #    # FIXME: Store complete reservoir or at least include the bias
            #    # FIXME: Doesn't work with too large reservoirs (>80 nodes)? This is because of the size limit of HDF5 headers
            #    #        Could be stored as dataset though...
            #    'r_weights': self.reservoir.w.todense(),
            #    'r_input'  : self.reservoir.w_in.todense()
            #    }
        )
        
        # in case of online experiments, episode number is taken from data file
        self.num_episode = int(self.collector.grp_name) + 1
    
    def __del__(self):
        del self.collector

