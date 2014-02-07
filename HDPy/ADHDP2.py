import numpy as np
from hdp import ADHDP

class ADHDP2(ADHDP):
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
        
        super(ADHDP2, self).__init__(reservoir, readout, *args, **kwargs)            
        
        self.reservoir = reservoir
        self.readout = readout
        self.withRecursion = _withRecursion
        self.with_bias = _withBias
        self.lambda_ = lambda_
        self.initial_action = initial_action
         
        self.countIterations = 0
         
        plant = kwargs['plant']
        action_space_dim = self.reservoir.get_input_dim() - plant.state_space_dim()
         
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
        self.w_bias_a = self.init_random_action_weights(action_space_dim, 1)
         
         
        # TODO: REMOVE THIS; FOR TESTING ONLY
        #self.withRecursion = True
         
        if self.withRecursion:
         
            # Initialize the action derivatives for the recursive case
            self.da_dWout_a_prec = np.zeros([action_space_dim, self.reservoir.output_dim])
            self.da_dWinout_aa_prec = np.zeros([action_space_dim, action_space_dim])
            self.da_dWinout_sa_prec = np.zeros([action_space_dim, self.reservoir.get_input_dim() - action_space_dim])
            self.da_dWout_a_curr = np.zeros([action_space_dim, self.reservoir.output_dim])
            self.da_dWinout_aa_curr = np.zeros([action_space_dim, action_space_dim])
            self.da_dWinout_sa_curr = np.zeros([action_space_dim, self.reservoir.get_input_dim() - action_space_dim])
             
            #Initialize the reservoir's states' derivatives for the recursive case
            self.dX_dWout_a_prec = np.zeros([action_space_dim, self.reservoir.output_dim])
            self.dX_dWinout_aa_prec = np.zeros([action_space_dim, action_space_dim])
            self.dX_dWinout_sa_prec = np.zeros([action_space_dim, self.reservoir.get_input_dim() - action_space_dim])
            self.dX_dWout_a_curr = np.zeros([action_space_dim, self.reservoir.output_dim])
            self.dX_dWinout_aa_curr = np.zeros([action_space_dim, action_space_dim])
            self.dX_dWinout_sa_curr = np.zeros([action_space_dim, self.reservoir.get_input_dim() - action_space_dim])
         
        self.action_space_dim = action_space_dim
         
        # Check assumptions
        assert self.reservoir.reset_states == False
        self.policy = kwargs['policy']
        #assert self.reservoir.get_input_dim() == self.policy.action_space_dim() + self.plant.state_space_dim()
        assert self.reservoir.get_input_dim() >= self.policy.initial_action().shape[0]      
        
    def init_random_action_weights(self, rows, cols):
        weight_matrice = np.zeros((rows, cols))
        for i in range(rows):
            weight_matrice[i][:] = np.random.normal(0.0, 0.1, cols)
        return weight_matrice 
    
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
        epoch = super(ADHDP2, self)._step(s_curr, epoch, a_curr, reward)     
        i_curr = epoch['i_curr']   
        i_next = epoch['i_next']
        x_curr = epoch['x_curr']
        x_next = epoch['x_next']
        a_curr = epoch['a_curr']
        a_next = epoch['a_next']
        j_curr = epoch['j_curr']
        j_next = epoch['j_next']
        deriv = epoch['deriv']
        err = epoch['err']
        
        # BEGIN ARTHUR
        
        #epoch = self.normalizer.normalize_epoch(epoch)
        
        action_space_dim = self.action_space_dim
        A = action_space_dim
        N = self.reservoir.w.shape[0]
         
        # Capture the different weights needed from the ESN output
        w_in_s = self.reservoir.w_in[:,:-action_space_dim]
        w_in_a = self.reservoir.w_in[:,-action_space_dim:]
        w_out_J = self.readout.beta[1:1+self.readout.input_dim-i_curr.shape[1]][:]
        w_inout_a = np.concatenate((self.w_inout_sa, self.w_inout_aa), axis=1)
        w_inout_aJ = self.readout.beta[-action_space_dim:][:].T
        weight_out_a = np.hstack((self.w_bias_a, self.w_out_a, self.w_inout_sa, self.w_inout_aa)).T
        #w_inout_aJ = self.w
                 
        # Get the different activations
        self.state_curr = self.reservoir.states                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        a = self.s_curr.values()
        
        self.sensor_curr = self.plant.state_input(epoch)
        
#         it = 0
#         for i in self.s_curr.keys():
#             if ("trg", "curr", "next", "deriv", "readout", "offline", "reward", "gamma"): continue
#             if (it >= self.sensor_curr.shape[1]): continue
#             self.sensor_curr[0][it] = self.normalizer.normalize_value(i, self.s_curr.get(i)[0])
#             it+=1
        self.action_curr = self.a_curr
        for i in range(self.a_curr.shape[0]):
            self.action_curr = self.normalizer.normalize_value('a_curr', self.a_curr[i])
        
        if not self.withRecursion:
            part_A = np.dot(w_out_J.T, np.atleast_2d(w_in_a.multiply(np.reshape(np.repeat(1-self.state_curr**2, A, axis=0), [N,A])))) + w_inout_aJ

            # Compute the derivatives:
            # Assuming no recursive dependency
            dJ_dW_bias_a = part_A
            dJ_dW_out_a = np.outer(part_A, self.state_curr)
            dJ_dW_inout_sa = np.outer(part_A, self.sensor_curr)
            dJ_dW_inout_aa = np.outer(part_A, self.action_curr)
            
#             dJ_dW_out_a = np.outer(part_A, self.state_curr) + w_inout_a * self.state_curr
#             dJ_dW_inout_sa = np.outer(part_A, self.sensor_curr) + w_inout_a * self.sensor_prec
#             dJ_dW_inout_aa = np.outer(part_A, self.action_curr) + w_inout_a * self.action_prec
        
        
        #BEGIN WITH RECURSION
        
        if self.withRecursion:
            
            # Transmit the old derivatives
            self.da_dWout_a_prec = self.da_dWout_a_curr
            self.da_dWinout_sa_prec = self.da_dWinout_sa_curr
            self.da_dWinout_aa_prec = self.da_dWinout_aa_curr
            
            self.dX_dWout_a_prec = self.dX_dWout_a_curr
            self.dX_dWinout_sa_prec = self.dX_dWinout_sa_curr
            self.dX_dWinout_aa_prec = self.dX_dWinout_sa_curr
            
            # Update the derivatives current values
            # TODO: CHECK DIMENSIONS AND MULTIPLIERS (outer, elementwise, dot, ...)
            # NOTE: (1-X**2) is the derivative of the tanh function, which is the activation function for this application
            self.dX_dWout_a_curr = (1 - self.state_curr**2) * (w_in_a * self.da_dWout_a_curr + self.reservoir.w * self.dX_dWout_a_prec)
            self.dX_dinWout_sa_curr = (1 - self.state_curr**2) * (w_in_a * self.da_dWinout_sa_curr + self.reservoir.w * self.dX_dWinout_sa_prec)
            self.dX_dinWout_aa_curr = (1 - self.state_curr**2) * (w_in_a * self.da_dWinout_aa_curr + self.reservoir.w * self.dX_dWinout_aa_prec)
            
            self.da_dWout_a_curr = self.state_curr + weight_out_a * self.dX_dWout_a_prec + w_inout_a * self.da_dWout_a_prec
            self.da_dWinout_sa_curr = self.sensor_curr + weight_out_a * self.dX_dWinout_sa_prec + w_inout_a * self.da_dWinout_sa_prec
            self.da_dWinout_aa_curr = self.action_curr + weight_out_a * self.dX_dWinout_aa_prec + w_inout_a * self.da_dWinout_aa_prec
            
            # Compute the Expected Return derivatives
            dJ_dW_out_a = w_out_J * self.dX_dWout_a_curr + w_inout_a * self.da_dWout_a_curr
            dJ_dW_inout_sa = w_out_J * self.dX_dWinout_sa_curr + w_inout_a * self.da_dWinout_sa_curr
            dJ_dW_inout_aa = w_out_J * self.dX_dWinout_aa_curr + w_inout_a * self.da_dWinout_aa_curr
        
        #END WITH RECURSION
        
        # Compute the deltas
        deltaW_bias_a = self.etha_w_out_a * dJ_dW_bias_a
        deltaW_out_a = self.etha_w_out_a * dJ_dW_out_a
        deltaW_inout_aa = self.etha_w_inout_aa * dJ_dW_inout_aa
        deltaW_inout_sa = self.etha_w_inout_sa * dJ_dW_inout_sa
        
        x_next1= np.concatenate((np.ones((x_next.shape[0], 1)), x_next), axis=1)
        
        # pre test
        a_pre_prop = np.tanh(x_next1.dot(weight_out_a)).T
        
        # Apply the deltas
        self.w_bias_a += deltaW_bias_a.T    
        self.w_out_a += deltaW_out_a
        self.w_inout_aa = self.w_inout_aa + deltaW_inout_aa
        self.w_inout_sa = self.w_inout_sa + deltaW_inout_sa
             
        # End of the current calculations, the states, sensor inputs and actions get old
        self.state_prec = self.state_curr
        self.sensor_prec = self.sensor_curr
        self.action_prec = self.action_curr
        
         
        # Let the activations flow through the network, and get the new action activation from that
        weight_out_a = np.hstack((self.w_bias_a, self.w_out_a, self.w_inout_sa, self.w_inout_aa)).T
        a_prop = np.tanh(x_next1.dot(weight_out_a)).T
#         scale = self.normalizer.get('a_curr')[1]
#         a_prop *= scale # Derivative denormalization
        
        proposer = BruteForceActor(self.reservoir, self.readout, self.plant)
        a_brute_prop = proposer.explore(epoch)
        self.countIterations += 1
        if self.countIterations % 100 == 0:
            print "Matthias': ", a_next
            print "Arthur's: ", a_prop
            print "Brute Force's: ", a_brute_prop
            print "   "
            
        
        #END ARTHUR
        
        # update epoch with locals:
        epoch['reward'] = np.atleast_2d([reward]).T
        epoch['deriv'] = deriv.T
        epoch['err'] = err.T
        epoch['readout'] = self.readout.beta.T
        epoch['gamma'] = np.atleast_2d([self.gamma(self.num_episode, self.num_step)]).T
        epoch['i_curr'] = i_curr
        epoch['x_curr'] = x_curr
        epoch['j_curr'] = j_curr
        epoch['a_curr'] = a_curr.T
        epoch['i_next'] = i_next
        epoch['x_next'] = x_next
        epoch['j_next'] = j_next
        epoch['a_next'] = a_next.T
        
        self._pre_increment_hook(epoch)
        
        # increment
        return epoch

class BruteForceActor:
    """Brute Force Decorator for the ADHDP2
    """
    def __init__(self, reservoir, readout, plant, *args, **kwargs):
        self.exploration_factor = [0.2, 0.1, -0.1, -0.2]
        self.reservoir = reservoir
        self.readout = readout
        self.plant = plant
    
    def _add_constant(self, x):
        """Add a constant term to the vector 'x'.
        x -> [1 x]
        """
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    
    def explore(self, epoch):
        """Execute one step of the actor and return the next action."""
        
        a_next = None
      
        #delta action
        a_delta = epoch['a_next'] - epoch['a_curr']
          
        # Next action
        a_next = epoch['a_next']
        j_best = epoch['j_next']
        in_state = self.plant.state_input(epoch)
          
        import itertools as it
        for exploration_factor in  it.product(self.exploration_factor, repeat=a_delta.shape[1]):
#                 candidate_nrm = self.normalizer.normalize_value('a_next', candidate)
            candidate = epoch['a_curr'] + (exploration_factor + a_delta)
            i_cand = np.vstack((in_state, candidate)).T
            x_cand = self.reservoir(i_cand, simulate=True)
            #x_cand = np.concatenate((x_cand, epoch['i_next']), axis=1)
            j_cand = self.readout(np.concatenate((x_cand, i_cand), axis=1))
            if j_cand > j_best:
                j_best = j_cand
                a_next = np.atleast_2d(candidate)
          
                #a_next = a_curr + self.alpha(self.num_episode, self.num_step) * (a_next - a_curr)
        
        return a_next
