"""
Assuming that all relevant values have been written into a HDF5 file
(e.g. using :py:class:`CollectingADHDP`), the intrinsics can be
analyzed. Here are some function to help with this.

It's assumed that a group represents one episode. There may be several
groups in a single HDF5 file, which would then be interpreted as several
runs (episodes) of the same experiment.

Primary interest:

- Sum of absolute readout weights: Information about converging
  behaviour of the readout. If the weights grow indefinitely, an
  overflow will eventually occur.
- Error: According to the theory, the error should converge to zero.
- Derivation per action: 
- Reward: 
- Accumulated reward per episode: 
- Neuron activation: 
- Action: 

Ideas:

* correctness of reward
* predicted return
  - At each step
  - Dependent on the action (maximum?)
* J, dJ and action, next action
* J_t vs. J_{t+1}, including action
* J(s,a)
* Maybe an 'animation' (characteristics over time) may be informative?

Plots from papers:

* reward over time; one line per episode, received and predicted
* utility distribution estimated by critic at a specific time step
* change of w_out over time
* utility and prediction (and raw sensor values) over time
* measured and predicted reward over time, one line per episode

If the controller is working, we should see that:

* If training is executed on always the same sequence of actions,
  the value prediction should become more accurate for earlier
  steps with increasing number of experiments.
  
  > This is since for estimating the TD-Error, we're relying on the current
    value to improve the estimate of the previous one.
    
  > More accurate = Error decreases

* If training is executed on always the same sequence of actions,
  the predicted value at time t should approximate the return of this
  action sequence (w.r.t. gamma)
  
  > The return prediction is the learning goal of the critic

* For several (known) action sequences, the predicted value of a state
  should roughly be the average, recursively computed return of all
  experiments which visited that state.
  
  > The return w.r.t. the policy is the average return over all action
    sequences

* The accumulated reward should increase with increasing episode

  > The return expresses the (expected) accumulated reward over time.
    The critic approximates the return
    If the policy converges to an optimal one, then the accumulated
    reward should increase with the number of episodes

* Any action besides the chosen one should yield a lower expected value

  > After training for some episodes, the approximation of the value
    should be reliable. Then, the policy should choose the next action
    optimally, meaning that the value of the next state must be maximal.

* The TD-Error should decrease with the number of episodes, if all
  states have been visited before.

  > The Critic is trained on the TD Error, such that it is minimized.
    With increasing training, the Error should converge to zero.

? The policy converges to an optimal policy; Given the estimate of the
  value dependent on the action, the next action should be close to the
  action maximizing the value function.


"""

import numpy as np
import pylab
import h5py

def gen_query(history):
    """To generate the old *query* function.
    
    .. deprecated:: 0.0
        Compatibility to the esn_acd history interface, also found in
        :py:meth:`Analysis.get_history`. Should not be used and will
        soon be removed.
    
    """
    query = lambda s: np.vstack([hep[s] for hep in history])
    return query

class Analysis:
    """
    """
    def __init__(self, pth):
        self.f = h5py.File(pth,'r')
        exp = map(int, self.f.keys())
        exp = sorted(exp)
        exp = map(str, exp)
        exp = filter(lambda k: len(self.f[k]) > 0, exp)
        self.experiments = exp
    
    def __del__(self):
        self.f.close()

    def stack_all_data(self):
        all_keys = self.f[self.experiments[0]].keys()
        data = {}
        for key in all_keys:
            data[key] = self.stack_data(key)
        
        return data
    
    def get_data(self, key):
        return [self.f[exp][key][:] for exp in self.experiments]
    
    def get_history(self):
        """Return the old *history* structure from the HDF5 data file.
        
        .. deprecated:: 0.0
            Compatibility to the esn_acd history interface. Should not
            be used and will soon be removed.
        
        """
        all_keys = self.f[self.experiments[0]].keys()
        history = {}
        for key in all_keys:
            history[k] = self.get_data(key)
        
        return history
    
    def stack_data(self, key):
        data = [self.f[exp][key][:] for exp in self.experiments]
        return np.concatenate(data)
    
    def num_steps(self, key=None):
        if key is None:
            key = self.f[self.experiments[0]].keys()[0]
        
        steps = [self.f[exp][key].shape[0] for exp in self.experiments]
        return steps
    
    def plot_grid(self, ax):
        steps = self.num_steps()
        for i in np.array(steps).cumsum():
            ax.axvline(i, linestyle='-', color='0.7')
    
    def plot_readout(self, ax):
        data = self.stack_data('readout')
        ax.plot(abs(data).sum(axis=1), 'k', label='Absolute Readout weight')
        ax.set_xlabel('step')
        ax.set_ylabel('Absolute Readout weight')
    
    def plot_readout_diff(self, ax):
        data = self.stack_data('readout')
        data = abs(data).sum(axis=1)
        ax.plot(data[1:] - data[:-1], 'k', label='Absolute Readout difference')
        ax.set_xlabel('step')
        ax.set_ylabel('Absolute Readout difference')
    
    def plot_reward(self, ax):
        data = self.stack_data('reward')
        ax.plot(data, 'k', label='Reward')
        ax.set_xlabel('step')
        ax.set_ylabel('Reward')
    
    def plot_derivative(self, ax):
        data = self.stack_data('deriv')
        ax.plot(data, 'k', label='Derivative')
        ax.set_xlabel('step')
        ax.set_ylabel('Derivative')
    
    def plot_actions(self, ax):
        a_curr = self.stack_data('a_curr')
        a_next = self.stack_data('a_next')
        ax.plot(a_curr, 'b', label='a_curr')
        ax.plot(a_next, 'r', label='a_next')
        ax.legend(loc=0)
        ax.set_xlabel('step')
        ax.set_ylabel('Action')
    
    def plot_error(self, ax):
        data = self.stack_data('err')
        ax.plot(data, 'k', label='error')
        ax.set_xlabel('step')
        ax.set_ylabel('TD-Error')
    
    def plot_accumulated_reward(self, ax):
        reward = self.get_data('reward')
        data = [r.sum() for r in reward]
        ax.plot(data, 'k', label='Accumulated reward')
        ax.set_xlabel('episode')
        ax.set_ylabel('Accumulated reward')

    def plot_path_return_prediction(self, ax, expno):
        """Plot the predicted return of a simulated path.
        
        The return prediction is specific to the path of the experiment
        *expno*. The plots will be written to *ax*.
        
        Five curves are shown:
        
        - Offline MSE
        - Online MSE (no bias)
        - Online MSE (with bias)
        - Predicted Value
        - Target
        
        For the first three curves, the respective method was trained
        on the path and then evaluated (thus displaying a training
        behaviour).
        The fourth curve shows the evaluation at the robot state of the
        estimated value function, based on the whole experiment. In
        contrast to the other curves, it incorporates all experiments
        up to *expno*. Specifically, note that early time steps may not
        reflect the target well, since the inspected path only counts
        towards the expected value but doesn't exclusively define it.
        The target is the discounted return for any step in the path
        of *expno*.
        
        .. todo::
            Online TD prediction
        
        """
        
        if expno not in self.experiments:
            raise Exception('expno is not a valid experiment id')
        
        x_curr = self.f[expno]['x_curr'][:]
        reward = self.f[expno]['reward'][:]
        gamma = self.f[expno]['gamma'][:]
        j_curr = self.f[expno]['j_curr'][:]
        reservoir_dim = x_curr.shape[1]
        N = x_curr.shape[0]
        
        trg = np.zeros((N,1))
        for i in range(N-2,-1,-1):
            trg[i] = trg[i+1] * gamma[i+1] + reward[i+1]
        
        trg[-1] = trg[-2]
        
        import mdp
        from rc import StabilizedRLS
        
        
        lr = mdp.nodes.LinearRegressionNode(input_dim=reservoir_dim,output_dim=1)
        lr.train(x_curr[:-1],trg[:-1])
        lr.stop_training()
        
        olrA = StabilizedRLS(with_bias=False, input_dim=reservoir_dim, output_dim=1, lambda_=0.9)
        olrB = StabilizedRLS(with_bias=True, input_dim=reservoir_dim, output_dim=1, lambda_=0.9)
        for i in range(N-1):
            s = np.atleast_2d(x_curr[i])
            t = np.atleast_2d(trg[i])
            olrA.train(s,t)
            olrB.train(s,t)
        
        ax.plot(lr(x_curr), 'b', label='Offline MSE')
        ax.plot(olrA(x_curr), 'c--', label='Online MSE (no bias)')
        ax.plot(olrB(x_curr), 'c:', label='Online MSE (bias)')
        ax.plot(j_curr, 'r', label='Predicted Value')
        ax.plot(trg, 'k', label='target')
        ax.legend(loc=0)
        ax.set_xlabel('step')
        ax.set_ylabel('return')
