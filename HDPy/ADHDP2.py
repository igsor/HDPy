import numpy as np
import hdp.ActorCritic as ActorCritic

class ADHDP2(ActorCritic):
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
    def __init__(self, reservoir, readout, initial_action=None, _withBias=True, etha=0.1, _withRecursion=False, lambda_=1.0, *args, **kwargs):
        self.reservoir = reservoir
        self.readout = readout
        self.withRecursion = _withRecursion
        self.with_bias = _withBias
        self.lambda_ = lambda_
        self.initial_action = initial_action
        
        action_space_dim = self.reservoir.get_action_space_dim()
        
        # Initialize the states
        self.state_prec = np.zeros([1, self.reservoir.states.shape[0]])
        self.state_curr = np.zeros([1, self.reservoir.states.shape[0]])
        self.sensor_prec = np.zeros([1, self.reservoir.get_input_dim() - action_space_dim])
        self.sensor_curr = np.zeros([1, self.reservoir.get_input_dim() - action_space_dim])
        self.action_prec = np.zeros([1, action_space_dim])
        self.action_curr = initial_action
        
        # Initialize the learning rates
        if etha is not None:
            self.etha_w_out_a = etha
            self.etha_w_inout_aa = etha
            self.etha_w_inout_sa = etha
        else:
            e = 0.1 # TODO: What is a good standard value?
            self.etha_w_out_a = e
            self.etha_w_inout_aa = e
            self.etha_w_inout_sa = e
        
        # Initialize the weight matrices for the action outputs
        self.w_out_a = self.init_random_action_weights(action_space_dim, self.reservoir.output_dim)
        self.w_inout_aa = self.init_random_action_weights(action_space_dim, action_space_dim)
        self.w_inout_sa = self.init_random_action_weights(action_space_dim, self.reservoir.get_input_dim() - action_space_dim)
        
        # Initialize the action derivatives for the recursive case
        self.da_dWout_a_prec = np.zeros(action_space_dim, self.reservoir.output_dim)
        self.da_dWinout_aa_prec = np.zeros(action_space_dim, action_space_dim)
        self.da_dWinout_sa_prec = np.zeros(action_space_dim, self.reservoir.get_input_dim() - action_space_dim)
        self.da_dWout_a_curr = np.zeros(action_space_dim, self.reservoir.output_dim)
        self.da_dWinout_aa_curr = np.zeros(action_space_dim, action_space_dim)
        self.da_dWinout_sa_curr = np.zeros(action_space_dim, self.reservoir.get_input_dim() - action_space_dim)
        
        #Initialize the reservoir's states' derivatives for the recursive case
        self.dX_dWout_a_prec = np.zeros(action_space_dim, self.reservoir.output_dim)
        self.dX_dWinout_aa_prec = np.zeros(action_space_dim, action_space_dim)
        self.dX_dWinout_sa_prec = np.zeros(action_space_dim, self.reservoir.get_input_dim() - action_space_dim)
        self.dX_dWout_a_curr = np.zeros(action_space_dim, self.reservoir.output_dim)
        self.dX_dWinout_aa_curr = np.zeros(action_space_dim, action_space_dim)
        self.dX_dWinout_sa_curr = np.zeros(action_space_dim, self.reservoir.get_input_dim() - action_space_dim)
            
        super(ADHDP2, self).__init__(*args, **kwargs)
        
        # Check assumptions
        assert self.reservoir.reset_states == False
        #assert self.reservoir.get_input_dim() == self.policy.action_space_dim() + self.plant.state_space_dim()
        assert self.reservoir.get_input_dim() >= self.policy.initial_action().shape[0]      
        
    def init_random_action_weights(self, rows, cols):
        weight_matrice = np.zeros((rows, cols))
        for i in range(rows):
            weight_matrice[i][:] = np.random.normal(0.0, 0.1, cols)
        return weight_matrice 
    
    def new_episode(self):
        """Start a new episode of the same experiment. This method can
        also be used to initialize the *ActorCritic*, for example when
        it is loaded from a file.
        """
        self.reservoir.reset()
        self.plant.reset()
        super(ADHDP2, self).new_episode()
    
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
        
        This is the baseline ADHDP algorithm. The next action is
        computed as
        
        .. math::
            a_{t+1} = m a_t + (1-m) \left( a_t + \\alpha \\frac{\\partial J(s_t, a_t)}{\\partial a} \\right)
        
        with :math:`m` the momentum and :math:`\\alpha` the step size.
        The critic trained on the TD error with discount rate
        :math:`\gamma`:
        
        .. math::
            err = r + \gamma J(s_{t+1}, a_{t+1}) - J(s_t, a_t)
        
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
        self.readout.trainAction(x_curr)
        
        
        
        # BEGIN ARTHUR
        
        action_space_dim = self.reservoir.get_action_space_dim()
        A = self.reservoir.get_action_space_dim()
        N = self.reservoir.w.shape[0]
         
        # Capture the different weights needed from the ESN output
        w_in_a = self.reservoir.w_in[:,-action_space_dim:]
        w_in_s = self.reservoir.w_in[:,:-action_space_dim]
        w_out_J = self.readout.beta[1:1+self.readout.input_dim-i_curr.shape[1]][:]
        w_inout_a = np.concatenate((self.w_inout_sa, self.w_inout_aa), axis=1)
        #w_inout_aJ = self.w
                 
        # Get the different activations
        self.state_curr = self.reservoir.states
        a = self.s_curr.values()
        it = 0
        for i in self.s_curr.keys():
            if "trg" in i: continue
            self.sensor_curr[0][it] = self.s_curr.get(i)[0]
            it+=1
        self.action_curr = self.a_curr
        
        # Compute the derivatives
        part_A = np.dot(w_out_J.T, np.atleast_2d(w_in_a.multiply(np.reshape(np.repeat(1-self.state_curr**2, A, axis=0), [N,A]))))# + winoutaj
        # Assuming no recursive dependency:
        dJ_dW_out_a = np.outer(part_A, self.state_curr) + w_inout_a * self.state_curr
        dJ_dW_inout_sa = np.outer(part_A, self.sensor_curr) + w_inout_a * self.sensor_prec
        dJ_dW_inout_aa = np.outer(part_A, self.action_curr) + w_inout_a * self.action_prec
         
        # Compute the deltas
        deltaW_out_a = self.etha_w_out_a * dJ_dW_out_a
        deltaW_inout_aa = self.etha_w_inout_aa * dJ_dW_inout_aa
        deltaW_inout_sa = self.etha_w_inout_sa * dJ_dW_inout_sa
        
        # pre test
        weight_out_a = np.hstack((self.w_out_a, self.w_inout_sa, self.w_inout_aa)).T
        a_pre_prop = np.tanh(x_next.dot(weight_out_a)).T
         
        # Apply the deltas
        self.w_out_a += deltaW_out_a
        self.w_inout_aa = self.w_inout_aa + deltaW_inout_aa
        self.w_inout_sa = self.w_inout_sa + deltaW_inout_sa
         
        # End of the current calculations, the states, sensor inputs and actions get old
        self.state_prec = self.state_curr
        self.sensor_prec = self.sensor_curr
        self.action_prec = self.action_curr
         
        # Let the activations flow through the network, and get the new action activation from that
        weight_out_a = np.hstack((self.w_out_a, self.w_inout_sa, self.w_inout_aa)).T
        a_prop = np.tanh(x_next.dot(weight_out_a)).T
        
        # END ARTHUR
        #BEGIN WITH RECURSION
        
        # TODO: PLACE THIS PART IN THE CODE, USE A BOOLEAN FOR ACTIVATING THE RECURSION
        
        self.da_dWout_a_prec = self.da_dWout_a_curr
        self.da_dWinout_sa_prec = self.da_dWinout_sa_curr
        self.da_dWinout_aa_prec = self.da_dWinout_aa_curr
        self.dX_dWout_a_prec = self.dX_dWout_a_curr
        self.dX_dWinout_sa_prec = self.dX_dWinout_sa_curr
        self.dX_dWinout_aa_prec = self.dX_dWinout_sa_curr
        
        # TODO: CHECK DIMENSIONS AND MULTIPLIERS (outer, elementwise, dot, ...)
        # NOTE: (1-X**2) is the derivative of the tanh function, which is the activation function for this application
        self.dX_dWout_a_curr = (1 - self.state_curr**2) * (w_in_a * self.da_dWout_a_curr + self.reservoir.w * self.dX_dWout_a_prec)
        self.dX_dinWout_sa_curr = (1 - self.state_curr**2) * (w_in_a * self.da_dWinout_sa_curr + self.reservoir.w * self.dX_dWinout_sa_prec)
        self.dX_dinWout_aa_curr = (1 - self.state_curr**2) * (w_in_a * self.da_dWinout_aa_curr + self.reservoir.w * self.dX_dWinout_aa_prec)
        
        self.da_dWout_a_curr = self.state_curr + weight_out_a * self.dX_dWout_a_prec + w_inout_a * self.da_dWout_a_prec
        self.da_dWinout_sa_curr = self.sensor_curr + weight_out_a * self.dX_dWinout_sa_prec + w_inout_a * self.da_dWinout_sa_prec
        self.da_dWinout_aa_curr = self.action_curr + weight_out_a * self.dX_dWinout_aa_prec + w_inout_a * self.da_dWinout_aa_prec
        
        dJ_dW_out_a = w_out_J * self.dX_dWout_a_curr + w_inout_a * self.da_dWout_a_curr
        dJ_dW_inout_sa = w_out_J * self.dX_dWinout_sa_curr + w_inout_a * self.da_dWinout_sa_curr
        dJ_dW_inout_aa = w_out_J * self.dX_dWinout_aa_curr + w_inout_a * self.da_dWinout_aa_curr
        
        #END WITH RECURSION
        
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
            i_curr, x_curr, j_curr = self._critic_eval(self.s_curr, self.a_curr, False, 'a_curr')
            self._pre_increment_hook(
                epoch,
                x_curr=x_curr,
                i_curr=i_curr,
                a_next=self.a_curr.T,
                a_curr=self.a_curr.T,
            )
        
        self.s_curr = epoch
        return self.policy.get_iterator(time_start_ms, time_end_ms, step_size_ms)
    
##ARTHUR
    
    
#     def __init__(self, input_dim, output_dim, res_size, etha, reservoir, with_bias=True, lambda_=1.0, with_recursion=False):
#         
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.lambda_ = lambda_
#         self.reservoir = reservoir
#         self._with_recursion = with_recursion
#         if etha is not None:
#             self.etha_w_outa = etha
#             self.etha_w_inoutaa = etha
#             self.etha_w_inoutsa = etha
#         else:
#             e = 0.5
#             self.etha_w_outa = e
#             self.etha_w_inoutaa = e
#             self.etha_w_inoutsa = e
#             
#         self.with_bias = with_bias
#         if self.with_bias:
#             input_dim += 1
#             
#         self._stop_training = False
#         self.w_out_a = np.zeros((output_dim-1, res_size)) #TODO: INIT RANDOMLY
#         self.prev_res_state = np.zeros((input_dim,1))
#         self.dX_dwouta = np.zeros((output_dim-1, res_size))
#         self.w_in_a = reservoir.w_in[:, -(reservoir.action_space_dim):] #TODO: NOT SURE ABOUT WHICH ONES
#         self.w_res = reservoir.w
#         #The neuronal activation function is typically a Sigmoid between -1 and 1 (a tanh function)
#         self.activationF = lambda x: np.tanh(x)
        
        
        
#     def train(self, w_out_J, sample, trg=None, err=None):
#         
#         if self.with_bias:
#             sample = self._add_constant(sample)
#         
#         #Compute the deltas for the output action weights
#         delta_w_out_a = np.zeros((self.input_dim, self.output_dim))
#         delta_w_inoutaa = np.zeros((self.input_dim, self.output_dim))
#         delta_w_inoutsa = np.zeros((self.input_dim, self.output_dim))
#         
#         for i in range(self.output_dim):
#             recursion_part = (self.dX_wouta * (self.w_in_a * self.w_out_a + self.w_res))
#             self.dX_dwouta = (np.ones((sample.shape[0], 1))-np.square(sample)).T.dot((self.w_in_a*self.prev_res_state) + recursion_part)
#             dJ_dwouta = w_out_J * self.dX_dwouta
#             delta_w_out_a[i] = self.etha_w_outa * dJ_dwouta
#             #Update the output action weights
#             self.w_out_a[i] += delta_w_out_a[i]
#         
#         self.prev_res_state = sample
#         
#     def _add_constant(self, x):
#         """Add a constant term to the vector 'x'.
#         x -> [1 x]
#         """
#         return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
#     
#     def stop_training(self):
#         """Disable filter adaption for future calls to ``train``."""
#         self._stop_training = True
        
##ARTHUR