"""Reinforcement learning parts of the rrl package. This mainly
includes the Actor-Critic-Design.

"""

import PuPy
import numpy as np
import cPickle as pickle

class Plant(object):
    """A template for Actor-Critic *plants*. The *Plant* describes the
    interaction of the Actor-Critic with the environment. Given a robot
    which follows a certain *Policy*, the environment generates rewards
    and robot states.
    
    """
    def __init__(self, state_space_dim=None):
        self._state_space_dim = state_space_dim
    
    def state_input(self, state, action):
        """Return the state-part of the critic input
        (i.e. the reservoir input).
        
        The state-part is derived from the current robot ``state`` and
        possibly also its ``action``. As return format, a Nx1 numpy
        vector is expected, where 2 dimensions should exist (e.g.
        :py:meth:`numpy.atleast_2d`).
        
        Although the reservoir input will consist of both, the *state*
        and *action*, this method must only return the *state* part of
        it.
        
        """
        raise NotImplementedError()
    
    def reward(self, epoch):
        """A reward generated by the *Plant* based on the current
        sensor readings in ``epoch``. The reward is single-dimensional.
        
        The reward is evaluated in every step. It builds the foundation
        of the approximated return.
        """
        raise NotImplementedError()
    
    def state_space_dim(self):
        """Return the dimension of the state space.
        This value is equal to the size of the vector returned by
        :py:meth:`state_input`.
        """
        if self._state_space_dim is None:
            raise NotImplementedError()
        return self._state_space_dim

class Policy(object):
    """A template for Actor-Critic *policies*. The *Policy* defines how
    an action is translated into a control (motor) signal. It
    continously receives action updates from the *Critic* which it has
    to digest.
    
    """
    def __init__(self, action_space_dim=None):
        self._action_space_dim = action_space_dim
    
    def initial_action(self):
        """Return the initial action. A valid action must be returned
        since the :py:class:`ActorCritic` relies on the format.
        
        The action has to be a 2-dimensional numpy vector, with both
        dimensions available.
        """
        raise NotImplementedError()
    
    def update(self, action_upd):
        """Update the *Policy* according to the current action update
        ``action_upd``, which was in turn computed by the
        :py:class:`ActorCritic`.
        """
        raise NotImplementedError()
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        """Return an iterator for the *motor_target* sequence, according
        to the current action configuration.
        
        The *motor_targets* glue the *Policy* and *Plant* together.
        Since they are applied in the robot and effect the sensor
        readouts, they are an "input" to the environment. As the targets
        are generated as effect of the action update, they are an output
        of the policy.
        
        """
        raise NotImplementedError()
    
    def action_space_dim(self):
        """Return the dimension of the action space.
        This value is equal to the size of the vector returned by
        :py:meth:`initial_action`.
        """
        if self._action_space_dim is None:
            raise NotImplementedError()
        return self._action_space_dim

class _ConstParam:
    """Stub for wrapping constant values into an executable function."""
    def __init__(self, value):
        self._value = value
    def __call__(self, time):
        """Return the constant value."""
        return self._value

class ADHDP(PuPy.PuppyActor):
    """Actor-critic design.
    
    The Actor-Critic estimates the return function
    
    .. math::
        J_t = \sum\limits_{k=t}^{T} \gamma^k r_{t+k+1}
    
    while the return is optimized at the same time. This is done by
    incrementally updating the estimate for :math:`J_t` and choosing
    the next action by optimizing the return in a single step. See
    [ESN-ACD]_ for details.
    
    ``reservoir``
        A reservoir instance compliant with the interface of
        :py:class:`SparseReservoirNode`. Specifically, must provide
        a *reset* method and *reset_states* must be :py:const:`False`.
        The input dimension must be compliant with the specification
        of the ``action``.
    
    ``readout``
        The reservoir readout function. An instance of ``PlainRLS`` is
        expected. Note that the readout must include a bias. The
        dimensions of ``reservoir`` and ``readout``  must match and the
        output of the latter must be single dimensional.
    
    ``plant``
        An instance of :py:class:`Plant`. The plant defines the
        interaction with the environment.
    
    ``policy``
        An instance of :py:class:`Policy`. The policy defines the
        interaction with the robot's actuators.
    
    ``gamma``
        Choice of *gamma* in the return function. May be a constant or
        a function of the time (relative to the episode start).
        
    ``alpha``
        Choice of *alpha* in the action update. May be a constant or a
        function of the time (relative to the episode start).
        
        The corresponding formula is
        
        .. math::
            a_{t+1} = a_{t} + \\alpha \\frac{\partial J_t}{\partial a_t}
        
        See [ESN-ACD]_ for details.
        
    """
    def __init__(self, reservoir, readout, plant, policy, gamma=1.0, alpha=1.0):
        super(ADHDP, self).__init__()
        self.reservoir = reservoir
        self.readout = readout
        self.plant = plant
        self.policy = policy
        self.set_alpha(alpha)
        self.set_gamma(gamma)
        self.num_episode = 0
        self.new_episode()
        
        # Check assumptions
        assert self.reservoir.reset_states == False
        assert self.reservoir.get_input_dim() == self.policy.action_space_dim() + self.plant.state_space_dim()
        assert self.readout.beta.shape == (self.reservoir.output_dim+1, 1)
        assert self.policy.initial_action().shape[0] >= 1
        assert self.policy.initial_action().shape[1] == 1
        assert self.reservoir.get_input_dim() >= self.policy.initial_action().shape[0]
    
    def new_episode(self):
        """Start a new episode of the same experiment. This method can
        also be used to initialize the *ActorCritic*, for example when
        it is loaded from a file.
        """
        self.reservoir.reset()
        self.num_episode += 1
        self.a_curr = self.policy.initial_action()
        self._motor_action_dim = self.policy.action_space_dim()
        self.s_curr = dict()
        self.num_step = 0
    
    def __call__(self, epoch, time_start_ms, time_end_ms, step_size_ms):
        """One round in the actor-critic cycle. The current observations
        are given in *epoch* and the timing information in the rest of
        the parameters. For a detailed description of the parameters,
        see :py:class:`PuPy.PuppyActor`.
        
        .. todo::
            Detailed description of the algorithm.
        
        """
        if self.num_step < 3: # FIXME: Initialization
            self.num_step += 1
            self.s_curr = epoch
            """ # FIXME: ESN-ACD comparison
            # phony history
            i = np.random.normal(size=(1, self.reservoir.get_input_dim()))
            x = np.random.normal(size=(1, self.reservoir.get_output_dim()))
            j = np.random.normal(size=(1,1))
            self._pre_increment_hook(
                epoch,
                np.random.normal(size=(1,1)),
                np.random.normal(size=(1,self._motor_action_dim)),
                np.random.normal(size=(1,1)),
                (i, x, j, self.a_curr),
                (i, x, j, self.a_curr)
                )
            #"""
            return self.policy.get_iterator(time_start_ms, time_end_ms, step_size_ms)
        
        # extern through the robot:
        # take action (a_curr = a_next in the previous run)
        # observe sensors values produced by the action (a_curr = previous a_next)
        
        # Generate reinforcement signal U(k), given in(k)
        reward = self.plant.reward(epoch)
        
        # ESN-critic, first instance: in(k) => J(k)
        in_state = self.plant.state_input(self.s_curr, self.a_curr)
        i_curr = np.vstack((in_state, self.a_curr)).T
        x_curr = self.reservoir(i_curr, simulate=False)
        j_curr = self.readout(x_curr)
        
        # Next action
        e = (np.ones(x_curr.shape) - x_curr**2).T # Nx1
        k = e * self.reservoir.w_in[:, -self._motor_action_dim:].toarray() # Nx1 .* NxA => NxA
        deriv = self.readout.beta[1:].T.dot(k) #  LxA
        deriv = deriv.T # AxL
        
        # gradient training of action (acc. to eq. 10)
        a_next = self.a_curr + self.alpha(self.num_step) * deriv
        #from math import pi # FIXME: ESN-ACD comparison
        #a_next = a_next % (2*pi) # FIXME: ESN-ACD comparison
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        in_state = self.plant.state_input(epoch, a_next)
        i_next = np.vstack((in_state, a_next)).T
        x_next = self.reservoir(i_next, simulate=True)
        j_next = self.readout(x_next)
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        self.readout.train(x_curr, e=err)
        
        # increment hook
        self._pre_increment_hook(
            epoch,
            reward,
            deriv,
            err,
            (i_curr, x_curr, j_curr, self.a_curr),
            (i_next, x_next, j_next, a_next)
            )
        
        # increment
        self.a_curr = a_next
        self.s_curr = epoch # TODO: Copy? # FIXME: Sufficient to store in_state for current epoch
        self.num_step += 1
        
        # return next action
        self.policy.update(a_next)
        return self.policy.get_iterator(time_start_ms, time_end_ms, step_size_ms)
    
    def _pre_increment_hook(self, epoch, reward, deriv, err, curr, next_):
        """Template method for subclasses.
        
        Before the actor-critic cycle increments, this method is invoked
        with all relevant locals of the :py:meth:`ActorCritic.__call__`
        method.
        """
        pass
    
    def save(self, pth):
        """Store the current instance in a file at ``pth``.
        
        .. note::
            If ``alpha`` or ``gamma`` was set to a user-defined
            function, make sure it's pickable. Especially, anonymous
            functions (:keyword:`lambda`) can't be pickled.
        
        """
        f = open(pth, 'w')
        pickle.dump(self, f)
        f.close()
    
    @staticmethod
    def load(pth):
        """Load an instance from a file ``pth``.
        """
        f = open(pth, 'r')
        cls = pickle.load(f)
        cls.new_episode()
        return cls
    
    def set_alpha(self, alpha):
        """Define a value for ``alpha``. May be either a constant or
        a function of the time.
        """
        if callable(alpha):
            self.alpha = alpha
        else:
            self.alpha = _ConstParam(alpha)
    
    def set_gamma(self, gamma):
        """Define a value for ``gamma``. May be either a constant or
        a function of the time.
        """
        if callable(gamma):
            self.gamma = gamma
        else:
            self.gamma = _ConstParam(gamma)

class CollectingADHDP(ADHDP):
    """Actor-Critic design with data collector.
    
    A :py:class:`PuPy.PuppyCollector` instance is created for recording
    sensor data and actor-critic internals together. The file is stored
    at ``expfile``.
    """
    def __init__(self, expfile, *args, **kwargs):
        self.expfile = expfile
        self.collector = None
        super(CollectingADHDP, self).__init__(*args, **kwargs)
    
    def _pre_increment_hook(self, epoch, reward, deriv, err, curr, next_):
        """Add the *ActorCritic* internals to the epoch and use the
        *collector* to save all the data.
        """
        epoch['reward']  = np.atleast_2d([reward]).T
        epoch['deriv']   = deriv.T
        epoch['err']     = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma']   = np.atleast_2d([self.gamma(self.num_step)]).T
        epoch['i_curr'], epoch['x_curr'], epoch['j_curr'], epoch['a_curr'] = curr
        epoch['i_next'], epoch['x_next'], epoch['j_next'], epoch['a_next'] = next_
        self.collector(epoch, 0, 1, 1)
    
    def save(self, pth):
        """Save the instance without the *collector*.
        
        .. note::
            When an instance is reloaded via :py:meth:`ActorCritic.load`
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
            headers={
                'r_weights': self.reservoir.w.todense(),
                'r_input'  : self.reservoir.w_in.todense()
                }
        )
    
    def __del__(self):
        del self.collector
