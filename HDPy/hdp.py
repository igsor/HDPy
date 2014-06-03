"""
On top of the :py:class:`ActorCritic` implementation, this module
provides a couple algorithms to solve a problem stated in terms of
Reinforcement Learning. All algorithms follow the same approach, namely
(action dependent) Heuristic Dynamic Programming. The baseline
algorithm is implemented in :py:class:`ADHDP`.

If a new specialisation of an :py:class:`ActorCritic` is created,
typically its :py:meth:`ActorCritic._step` method is adapted (this is
for example the case in :py:class:`ADHDP`). If so, the two methods
:py:meth:`ActorCritic._pre_increment_hook` (before returning) and
:py:meth:`ActorCritic._next_action_hook` (after computation of the
next action) should be called as other structures may rely on those.

Some variations of the baseline algorithm are implemented as well in
:py:class:`ActionGradient`, :py:class:`ActionRecomputation` and
:py:class:`ActionBruteForce`. They fulfill the same purpose but approach
it differently (specifically, the actor is implemented differently). The
details are given in details of the respective class.

"""
import numpy as np
import PuPy
from rl import ActorCritic
import warnings


class ADHDP(ActorCritic):
    """Action dependent Heuristic Dynamic Programming structure and
    basic algorithm implementation.
    
    In the :py:meth:`_step` method, this class provides the
    implementation of a baseline algorithm. By default, the behaviour
    is online, i.e. the critic is trained and the actor in effect.
    Note that the actor can be modified through the
    :py:meth:`_next_action_hook` routine.
    
    ``reservoir``
        A reservoir instance compliant with the interface of
        :py:class:`ReservoirNode`. Specifically, must provide
        a *reset* method and *reset_states* must be :py:const:`False`.
        The input dimension must be compliant with the specification
        of the ``action``.
    
    ``readout``
        The reservoir readout function. Usually an online linear
        regression (RLS) instance, like :py:class:`StabilizedRLS`.
        Note that the readout must include a bias. The dimensions
        of ``reservoir`` and ``readout``  must match and the
        output of the latter must be single dimensional.
    
    ``iomodel``
        :py:type:bool: whether or not to use the I/O-model, where the
        readout's input is the reservoir states plus the reservoir inputs.
    
    """
    def __init__(self, reservoir, readout, iomodel=True, train=True, *args, **kwargs):
        self.reservoir = reservoir
        self.readout = readout
        self.iomodel = iomodel
        self.train = train
        super(ADHDP, self).__init__(*args, **kwargs)
        
        # Check assumptions
        assert self.reservoir.reset_states == False
        assert self.reservoir.get_input_dim() == self.child.action_space_dim() + self.plant.state_space_dim()
        assert self.reservoir.get_input_dim() >= self.child.initial_action().shape[0]
    
    def new_episode(self):
        """Start a new episode of the same experiment. This method can
        also be used to initialize the *ActorCritic*, for example when
        it is loaded from a file.
        """
        self.reservoir.reset()
        super(ADHDP, self).new_episode()
    
    def _critic_eval(self, state, action, simulate, action_name='a_curr'):
        """Evaluate the critic at ``state`` and ``action``."""
        in_state = self.plant.state_input(state)
        action_nrm = self.normalizer.normalize_value(action_name, action)
        # to concatenate state and action, the action needs to be expanded to epoch length (in case they don't match):
        r_input = np.vstack((in_state, np.repeat(action_nrm, in_state.shape[1], axis=1))).T
        #r_input += np.random.normal(scale=0.001, size=r_input.shape)
        r_state = self.reservoir(r_input, simulate=simulate)
        if self.iomodel:
            r_state = np.hstack((r_state, r_input)) # I/O model
        # for the readout we only take the last step:
        j_curr = self.readout(r_state[np.newaxis, -1])
        return r_input, r_state, j_curr
        
    def _critic_deriv(self, r_state):
        """Return the critic's derivative at ``r_state``."""
        if self.iomodel:
            return self._critic_deriv_io_model(r_state)
        else:
            return self._critic_deriv_direct_model(r_state)
    
    def _critic_deriv_io_model(self, r_state):
        """Return the critic's derivative at ``r_state``."""
        direct_input_size = self.plant.state_space_dim()+self.child.action_space_dim() # Input/Output ESN Model
        r_state = r_state[:, :-direct_input_size] # this is because _critic_eval appends the input to the state
        dtanh = (np.ones(r_state.shape) - r_state**2).T # Nx1
        dstate = dtanh * self.reservoir.w_in[:, -self._action_space_dim:].toarray() # Nx1 .* NxA => NxA
        deriv = self.readout.beta[1:-direct_input_size].T.dot(dstate) # Input/Output ESN Model
        deriv += self.readout.beta[-self._action_space_dim:].T # Input/Output ESN Model
        deriv = deriv.T # AxL
        scale = self.normalizer.get('a_curr')[1]
        deriv *= scale # Derivative denormalization
        return deriv
    
    def _critic_deriv_direct_model(self, r_state):
        """Return the critic's derivative at ``r_state``."""
        dtanh = (np.ones(r_state.shape) - r_state**2).T # Nx1
        dstate = dtanh * self.reservoir.w_in[:, -self._action_space_dim:].toarray() # Nx1 .* NxA => NxA
        deriv = self.readout.beta[1:].T.dot(dstate) #  LxA # Direct ESN Model
        deriv = deriv.T # AxL
        scale = self.normalizer.get('a_curr')[1]
        deriv *= scale # Derivative denormalization
        return deriv
    
    def _step(self, s_curr, epoch, a_curr, reward):
        """Execute one step of the actor and return the next action.
        
        This is the baseline ADHDP algorithm. The next action is
        computed as
        
        .. math::
            a_{t+1} = m a_t + (1-m) \left( a_t + \\alpha \\frac{\\partial J(s_t, a_t)}{\\partial a} \\right)
        
        with :math:`m` the momentum and :math:`\\alpha` the step size.
        The critic trained on the TD error with discount rate
        :math:`\gamma`:
        
        .. math::
            err = r_{t+1} + \gamma J(s_{t+1}, a_{t+1}) - J(s_t, a_t)
        
        ``epoch``
            Latest observed state (s_next). :py:keyword:`dict`, same as ``epoch``
            of the :py:meth:`__call__`.
        
        ``s_curr``
            Previous observed state. :py:keyword:`dict`, same as ``s_curr``
            of the :py:meth:`__call__`.
        
        ``reward``
            Reward of ``epoch``
        
        """
        # ESN-critic, first instance: in(k) => J(k)
        i_curr, x_curr, j_curr = self._critic_eval(s_curr, a_curr, simulate=False, action_name='a_curr')
        
        # Next action
        deriv = self._critic_deriv(x_curr[np.newaxis, -1])
        
        # gradient training of action (acc. to eq. 10)
        a_next = a_curr + self.alpha(self.num_episode, self.num_step) * deriv
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, x_next, j_next = self._critic_eval(epoch, a_next, simulate=True, action_name='a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        if self.train:
            self.readout.train(x_curr[np.newaxis, -1], err=err)
#            self.readout._stop_training() ## !!!! Nico: Hack for mdp-LinearRegressionNode to work here! !!!! XXXX
        
        # TODO: make optional list with locals to store in epoch
        # update epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['deriv'] = deriv.T
        epoch['err'] = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['x_curr'] = x_curr
        epoch['j_curr'] = j_curr
        epoch['i_next'] = i_next
        epoch['x_next'] = x_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        # increment
        return epoch
    
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
        epoch_raw = epoch.copy()
        if self.num_step > 1:
            i_curr, x_curr, j_curr = self._critic_eval(self.s_curr, self.a_curr, False, 'a_curr')
            epoch['x_curr'] = x_curr
            epoch['i_curr'] = i_curr
        
        epoch['a_curr'] = self.a_curr.T
        epoch['a_next'] = self.a_curr.T
        self.s_curr = epoch_raw
        return self.child(epoch, time_start_ms, time_end_ms, step_size_ms)

## HDP VARIATIONS ##


class LinearADHDP(ActorCritic):
    def __init__(self, readout, train=True, *args, **kwargs):
        self.readout = readout
        self.train = train
        super(LinearADHDP, self).__init__(*args, **kwargs)
    
    def _critic_eval(self, state, action, action_name='a_curr'):
        """Evaluate the critic at ``state`` and ``action``."""        
        in_state = self.plant.state_input(state)
        action_nrm = self.normalizer.normalize_value(action_name, action)
        i_curr = np.vstack((in_state, np.repeat(action_nrm, in_state.shape[1], axis=1))).T
        j_curr = self.readout(i_curr[np.newaxis, -1])
        
        return i_curr, j_curr
    
    def _step(self, s_curr, epoch, a_curr, reward):
        # ESN-critic, first instance: in(k) => J(k)
        i_curr, j_curr = self._critic_eval(s_curr, a_curr, action_name='a_curr')
        
        # Next action
        deriv = self.readout.beta[-self._action_space_dim:] # Input/Output ESN Model
        scale = self.normalizer.get('a_curr')[1]
        deriv *= scale # Derivative denormalization
        
        # gradient training of action (acc. to eq. 10)
        a_next = a_curr + self.alpha(self.num_episode, self.num_step) * deriv
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, j_next = self._critic_eval(epoch, a_next, action_name='a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        if self.train:
            self.readout.train(i_curr[np.newaxis, -1], err=err)
        
        # TODO: make optional list with locals to store in epoch
        # update epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['deriv'] = deriv.T
        epoch['err'] = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['j_curr'] = j_curr
        epoch['i_next'] = i_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        # increment
        return epoch
    
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
        epoch_raw = epoch.copy()
        if self.num_step > 1:
            i_curr, j_curr = self._critic_eval(self.s_curr, self.a_curr, 'a_curr')
            epoch['i_curr'] = i_curr
        
        epoch['a_curr'] = self.a_curr.T
        epoch['a_next'] = self.a_curr.T
        self.s_curr = epoch_raw
        return self.child(epoch, time_start_ms, time_end_ms, step_size_ms)

class ReservoirActorADHDP(ADHDP):
    """The next action is determined by a separate readout of the reservoir.
    The action-readout weights are trained by gradient descending
    
    """
    def __init__(self, *args, **kwargs):
        self.gd_tol = kwargs.pop('gd_tolerance', 1e-15)
        self.gd_max_iter = kwargs.pop('gd_max_iter', 500)
        self.rho, self.beta = 0.001, 0.1
        self.alpha_max = 20.0
        self.ls_max_iter = 1000
        super(ActionGradient, self).__init__(*args, **kwargs)
    
    def _step(self, s_curr, epoch, a_curr, reward):
        """Execute one step of the actor and return the next action."""
        # ESN-critic, first instance: in(k) => J(k)
        in_state = self.plant.state_input(s_curr)
        action_nrm = self.normalizer.normalize_value('a_curr', a_curr)
        i_curr = np.vstack((in_state, action_nrm)).T
        r_prev = self.reservoir.states
        r_state = self.reservoir(i_curr, simulate=False)
        x_curr = np.hstack((r_state, i_curr)) # TODO: Input/Output ESN Model
        j_curr = self.readout(x_curr)
        
        # Gradient ascent of J(a|s_{t+1})
        direct_input_size = self.plant.state_space_dim()+self.child.action_space_dim() # Input/Output ESN Model
        dtanh = (np.ones(r_state.shape) - r_state**2).T # Nx1
        dstate = dtanh * self.reservoir.w_in[:, -self._action_space_dim:].toarray() # Nx1 .* NxA => NxA
        deriv = self.readout.beta[1:-direct_input_size].T.dot(dstate) # Input/Output ESN Model
        deriv += self.readout.beta[-self._action_space_dim:].T # Input/Output ESN Model
        deriv = deriv.T # AxL
        scale = self.normalizer.get('a_curr')[1]
        deriv *= scale # Derivative denormalization
        #return deriv
        
        
        a_next = self.gradient_descent(epoch, a_curr)
        deriv = (a_next - a_curr) / self.alpha(self.num_episode, self.num_step)
        
        # Next action
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, x_next, j_next = self._critic_eval(epoch, a_next, simulate=True, action_name='a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        self.readout.train(x_curr, err=err) 
        
        # fill epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['deriv'] = deriv.T
        epoch['err'] = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['x_curr'] = x_curr
        epoch['j_curr'] = j_curr
        epoch['i_next'] = i_next
        epoch['x_next'] = x_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        # increment
        return epoch

class ActionGradient(ADHDP):
    """Determine the next action by gradient ascent search.
    The gradient ascent computes the action which maximizes the
    predicted return for a fixed state.
    
    Additional keyword arguments:
    
    ``gd_tolerance``
        Stop gradient descent if gradient below this threshold.
    
    ``gd_max_iter``
        Maximum number of gradient descent steps
    
    """
    def __init__(self, *args, **kwargs):
        self.gd_tol = kwargs.pop('gd_tolerance', 1e-15)
        self.gd_max_iter = kwargs.pop('gd_max_iter', 500)
        self.rho, self.beta = 0.001, 0.1
        self.alpha_max = 20.0
        self.ls_max_iter = 1000
        super(ActionGradient, self).__init__(*args, **kwargs)
    
    def _step(self, s_curr, epoch, a_curr, reward):
        """Execute one step of the actor and return the next action."""
        # ESN-critic, first instance: in(k) => J(k)
        i_curr, x_curr, j_curr = self._critic_eval(s_curr, a_curr, simulate=False, action_name='a_curr')
        
        # Gradient ascent of J(a|s_{t+1})
        a_next = self.gradient_descent(epoch, a_curr)
        deriv = (a_next - a_curr) / self.alpha(self.num_episode, self.num_step)
        
        # Next action
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, x_next, j_next = self._critic_eval(epoch, a_next, simulate=True, action_name='a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        self.readout.train(x_curr, err=err)
        
        # fill epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['deriv'] = deriv.T
        epoch['err'] = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['x_curr'] = x_curr
        epoch['j_curr'] = j_curr
        epoch['i_next'] = i_next
        epoch['x_next'] = x_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        # increment
        return epoch
    
    def _line_search(self, state, action, gradient):
        """Line search algorithm to compute the step size parameter for
        a gradient step at ``action`` into direction ``gradient``.
        The ``state`` is still fixed, but must be provided.
        
        .. todo::
            broken for some cases (no reduction, infinite loop)
        
        """
        #0 < rho  < 0.5
        #rho < beta < 1.0
        warnings.warn('This code is unreliable. Use a fixed step size instead.')
        
        def phi(alpha):
            """step-size centric gradient step representation."""
            action_query = action + alpha * gradient
            query_result = self._critic_eval(state, action_query, simulate=True, action_name='a_curr') # TODO: Check reservoir state!
            return - query_result[-1]
        
        def dphi(alpha):
            """step-size centric gradient step representation, first
            derivative."""
            # gradient * -derivative(action + alpha * gradient)
            action_query = action + alpha * gradient
            in_state = self.plant.state_input(state)
            action_nrm = self.normalizer.normalize_value('a_next', action_query)
            i_inter = np.vstack((in_state, action_nrm)).T
            x_inter = self.reservoir(i_inter, simulate=True) # TODO: Check reservoir state!
            deriv = self._critic_deriv(x_inter)
            return gradient * -deriv
        
        dphi_zero, phi_zero = dphi(0.0), phi(0.0)
        lambda_ = lambda al: phi_zero + self.rho * al * dphi_zero
        
        if (dphi_zero >= 0.0).any():
            #warnings.warn('step_size cannot increase the function')
            print 'WARNING: step_size cannot increase the function (dphi(0.0) = %f)' % dphi_zero
            return 0.0
        
        # enforce lower bound
        gamma = self.beta * dphi_zero
        a = 0
        b = min(1.0, self.alpha_max)
        num_iter = self.ls_max_iter
        while b < self.alpha_max and (phi(b) <= lambda_(b)).any() and (dphi(b) <= gamma).any() and num_iter > 0:
            #print a,b
            a = b
            b = min(2*b, self.alpha_max)
            num_iter -= 1
        
        # set alpha
        alpha = b
        
        # enforce both constraints
        # infinite loop possible! how & where??
        num_iter = self.ls_max_iter
        while ((phi(alpha) > lambda_(alpha)).any() or (dphi(alpha) < gamma).any()) and num_iter > 0:
            alpha, a, b = self._refine( (a, b), phi, dphi, lambda_)
            if abs(a - b) < 1e-14:
                break
            num_iter -= 1
        
        if (phi(alpha) >= phi_zero).any():
            #warnings.warn('step size does not increase the function')
            print 'WARNING: step size does not increase the function (phi(alpha)=%f)' % phi(alpha)
            alpha = 0.0
        
        return alpha
    
    def _refine(self, (a, b), phi, dphi, lambda_):
        """Refinement part of the line search algorithm. Adjusts
        alpha and narrows the interval [a,b].
        """
        D = b - a
        if D < 1e-15:
            print "WARNING: a ~= b (D=", D
            #return a, a, b
        
        c = (phi(b) - phi(a) - D * dphi(a) ) / D**2
        
        # compute alpha
        if (c > 0).any():
            alpha = a - dphi(a) / (2*c)
            alpha = min(max(alpha, a + 0.1 * D), b - 0.1 * D)
        else:
            alpha = (a+b)/2
        
        # adjust interval
        if (phi(alpha) < lambda_(alpha)).any():
            a = alpha
        else:
            b = alpha
        
        return alpha, a, b
    
    def gradient_descent(self, state, action):
        """Gradient ascent search to find the action which maximizes
        the predicted return given a ``state``. The search starts at
        ``action``.
        """
        in_state = self.plant.state_input(state)
        num_iter = self.gd_max_iter
        while True:
            # Compute the gradient
            action_nrm = self.normalizer.normalize_value('a_next', action)
            i_inter = np.vstack((in_state, action_nrm)).T
            x_inter = self.reservoir(i_inter, simulate=True)
            x_inter = np.hstack((x_inter, i_inter)) # FIXME: Input/Output ESN Model
            gradient = self._critic_deriv(x_inter)
            
            # Do line search and update the action
            #step_size = self._line_search(state, action, gradient)
            step_size = self.alpha(self.num_episode, self.num_step)
            action = action + step_size * gradient
            action = self._next_action_hook(action)
            
            # Exit condition
            num_iter -= 1
            if np.linalg.norm(gradient) < self.gd_tol or num_iter <= 0:
                break
        
        #print num_iter
        
        return action

class ActionRecomputation(ADHDP):
    """Determine the next action the same way as the baseline algorithm
    for critic training, then recompute it based on the updated critic
    and with the latest state information.
    
    """
    def _step(self, s_curr, epoch, a_curr, reward):
        """Execute one step of the actor and return the next action."""
        # ESN-critic, first instance: in(k) => J(k)
        i_curr, x_curr, j_curr = self._critic_eval(s_curr, a_curr, False, 'a_curr')
        
        # Next action
        deriv = self._critic_deriv(x_curr)
        
        # gradient training of action (acc. to eq. 10)
        a_next = a_curr + self.alpha(self.num_episode, self.num_step) * deriv
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, x_next, j_next = self._critic_eval(epoch, a_next, True, 'a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        self.readout.train(x_curr, err=err)
        
        # Action re-computation
        deriv2 = self._critic_deriv(x_next)
        a_next = a_curr + self.alpha(self.num_episode, self.num_step) * deriv2
        a_next = self.momentum(a_curr, a_next, self.num_episode, self.num_step)
        a_next = self._next_action_hook(a_next)
        
        # fill epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['deriv2'] = deriv2.T
        epoch['deriv'] = deriv.T
        epoch['err'] = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['x_curr'] = x_curr
        epoch['j_curr'] = j_curr
        epoch['i_next'] = i_next
        epoch['x_next'] = x_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        # increment
        return epoch

class ActionBruteForce(ADHDP):
    """Find the optimal action by computing the expected return at
    different sampled locations and picking the action which yields the
    highest one.
    
    ``candidates``
        Action samples. Must be a list of valid actions.
    
    .. todo::
        Breaks old code
    
    """
    def __init__(self, candidates, *args, **kwargs):
        super(ActionBruteForce, self).__init__(*args, **kwargs)
        self.candidates = candidates
    
    def _step(self, s_curr, epoch, a_curr, reward):
        """Execute one step of the actor and return the next action."""
        # ESN-critic, first instance: in(k) => J(k)
        i_curr, x_curr, j_curr = self._critic_eval(s_curr, a_curr, False, 'a_curr')
        
        # Next action
        a_next = a_curr
        j_best = float('-inf')
        in_state = self.plant.state_input(epoch)
        for candidate in self.candidates:
            candidate_nrm = self.normalizer.normalize_value('a_next', candidate)
            i_cand = np.vstack((in_state, candidate_nrm)).T
            x_cand = self.reservoir(i_cand, simulate=True)
            j_cand = self.readout(x_cand)
            if j_cand > j_best:
                j_best = j_cand
                a_next = np.atleast_2d(candidate)
        
        a_next = a_curr + self.alpha(self.num_episode, self.num_step) * (a_next - a_curr)
        a_next = self._next_action_hook(a_next)
        
        # ESN-critic, second instance: in(k+1) => J(k+1)
        i_next, x_next, j_next = self._critic_eval(epoch, a_next, True, 'a_next')
        
        # TD_error(k) = J(k) - U(k) - gamma * J(k+1)
        err = reward + self.gamma(self.num_episode, self.num_step) * j_next - j_curr
        
        # One-step RLS training => Trained ESN
        self.readout.train(x_curr, err=err)
        
        # fill epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['err'] = err.T
        epoch['deriv'] = a_next-a_curr
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['x_curr'] = x_curr
        epoch['j_curr'] = j_curr
        epoch['i_next'] = i_next
        epoch['x_next'] = x_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        # increment
        return epoch
