"""
Assuming that all relevant values have been written into a HDF5 file
(e.g. using :py:class:`CollectingADHDP`), the intrinsics can be
analyzed. Here are some function to help with this.

It's assumed that a group represents one episode. There may be several
groups in a single HDF5 file, which would then be interpreted as several
runs (episodes) of the same experiment.

Quick plots
~~~~~~~~~~~

- Sum of absolute readout weights: Information about converging
  behaviour of the readout. If the weights grow indefinitely, an
  overflow will eventually occur.

- Error: According to the theory, the error should converge to zero.

- Derivation per action: Action change

- Reward: Desired to converte to the maximum (in the long run)

- Accumulated reward per episode: Desired to converge to the maximum

- Neuron activation: ?

- Action: If the policy converges, the action should converge over several episodes

If the controller is working, we should see that
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If training is executed on always the same sequence of actions,
  the value prediction should become more accurate for earlier
  steps with increasing number of experiments.
  
  > This is since for estimating the TD-Error, we're relying on the current
    value to improve the estimate of the previous one.
    
  > More accurate = Error decreases
  
  ! ``plot_error_over_episodes``

* If training is executed on always the same sequence of actions,
  the predicted value at time t should approximate the return of this
  action sequence (w.r.t. gamma)
  
  > The return prediction is the learning goal of the critic
  
  ! ``plot_predicted_return_over_episodes``

* The accumulated reward should increase with increasing episode

  > The return expresses the (expected) accumulated reward over time.
    The critic approximates the return
    If the policy converges to an optimal one, then the accumulated
    reward should increase with the number of episodes
  
  ! ``plot_reward``

* For several (known) action sequences, the predicted value of a state
  should roughly be the average, recursively computed return of all
  experiments which visited that state.
  
  > The return w.r.t. the policy is the average return over all action
    sequences
  
  ! Not implemented

* Any action besides the chosen one should yield a lower expected value

  > After training for some episodes, the approximation of the value
    should be reliable. Then, the policy should choose the next action
    optimally, meaning that the value of the next state must be maximal.
  
  ! Not implemented

* The TD-Error should decrease with the number of episodes, if all
  states have been visited before.

  > The Critic is trained on the TD Error, such that it is minimized.
    With increasing training, the Error should converge to zero.
  
  ! Not implemented

? The policy converges to an optimal policy; Given the estimate of the
  value dependent on the action, the next action should be close to the
  action maximizing the value function.
  
  ! Not implemented

Plots from papers
~~~~~~~~~~~~~~~~~

* reward over time; one line per episode, received and predicted

  > When the predicted return (?) is shown, it can be seen that (assuming
    all works well) it propagates towards earlier timesteps in the episode
    (assuming several runs with the same or similar action).

* utility distribution estimated by critic at a specific time step

  ? Not sure, how to interpret this (generally)

* change of w_out over time

  > Shows the effect of the critic learning

* utility and prediction (and raw sensor values) over time

  > The papers suggest that the prediction should display the form of the
    utility a couple of timesteps earlier.


Other ideas
~~~~~~~~~~~

* Show action selection fitness

  > In every step, plot J(a|s), J(a|s_{t+1}). Also indicate a_t and a_{t+1} with vertical bars
  
  > At least with increasing episode count, the action a_{t+1} should be
    close to the optimum of J(a|s_{t+1})

  ? Maybe also annotate or plot the gradient

* Given all paths of an experiment:

  - Can compute the return of each path
  - Can average return of a state, given all observed paths
  - Can compare the average return of a state with the TD-predicted return (evaluating the critic for a sequence)
  - Comparison is based on sequences

? Correctness of reward

? Predicted return

? Maybe an 'animation' (characteristics over time) may be informative

* Reservoir inputs; Goes together with reservoir output (neuron potential)

"""

import numpy as np
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
    """Collection of functions to analyze a HDF5 data file at ``pth``.
    """
    def __init__(self, pth, grid=False):
        self.f = h5py.File(pth,'r')
        exp = map(int, self.f.keys())
        exp = sorted(exp)
        exp = map(str, exp)
        exp = filter(lambda k: len(self.f[k]) > 0, exp)
        self.experiments = exp
        self.always_plot_grid = grid
    
    def __del__(self):
        """Close open files."""
        self.f.close()

    def stack_all_data(self):
        """Return a :py:keyword:`dict` with all concatenated data,
        sorted by the data key."""
        all_keys = self.f[self.experiments[0]].keys()
        data = {}
        for key in all_keys:
            data[key] = self.stack_data(key)
        
        return data
    
    def get_data(self, key):
        """Return a list of all data belonging to ``key``."""
        return [self.f[exp][key][:] for exp in self.experiments]
    
    def get_history(self):
        """Return the old ``history`` structure from the HDF5 data file.
        
        .. deprecated:: 0.0
            Compatibility to the esn_acd history interface. Should not
            be used and will soon be removed.
        
        """
        all_keys = self.f[self.experiments[0]].keys()
        history = {}
        for key in all_keys:
            history[key] = self.get_data(key)
        
        return history
    
    def stack_data(self, key):
        """Return data related to ``key`` of all experiments in a single
        array."""
        data = [self.f[exp][key][:] for exp in self.experiments]
        return np.concatenate(data)
    
    def num_steps(self, key=None):
        """Return the number of steps per experiment."""
        if key is None:
            key = self.f[self.experiments[0]].keys()[0]
        
        steps = [self.f[exp][key].shape[0] for exp in self.experiments]
        return steps
    
    def plot_grid(self, axis):
        """Add a vertical bar to ``axis`` to mark experiment
        boundaries."""
        steps = self.num_steps()
        for i in np.array(steps).cumsum():
            axis.axvline(i, linestyle='-', color='0.7')
        
        return axis
    
    def plot_neuron_activation(self, axis):
        """Plot the absolute sum of the neuron state (=activation
        potential) of the current (red) and next (blue) timestep in
        ``axis``."""
        x_curr = self.stack_data('x_curr')
        x_next = self.stack_data('x_next')
        axis.plot(abs(x_curr).sum(axis=1), 'r', label='Absolute Neuron Activation')
        axis.plot(abs(x_next).sum(axis=1), 'b', label='Absolute Neuron Activation')
        axis.set_xlabel('step')
        axis.set_ylabel('Absolute Neuron Activation')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return ax
    
    def plot_readout(self, axis):
        """Plot the absolute readout weight over time in ``axis``."""
        data = self.stack_data('readout')
        axis.plot(abs(data).sum(axis=1), 'k', label='Absolute Readout Weights')
        axis.set_xlabel('step')
        axis.set_ylabel('Absolute Readout Weights')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_readout_diff(self, axis):
        """Plot the difference of absolute readout weights in
        ``axis``."""
        data = self.stack_data('readout')
        data = abs(data).sum(axis=1)
        axis.plot(data[1:] - data[:-1], 'k', label='Absolute Readout difference')
        axis.set_xlabel('step')
        axis.set_ylabel('Absolute Readout difference')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_reward(self, axis):
        """Plot the reward over time in ``axis``."""
        data = self.stack_data('reward')
        axis.plot(data, 'k', label='Reward')
        axis.set_xlabel('step')
        axis.set_ylabel('Reward')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_derivative(self, axis):
        """Plot the derivative dJ/da over time in ``axis``."""
        data = self.stack_data('deriv')
        axis.plot(data, 'k', label='dJ(k)/da')
        axis.set_xlabel('step')
        axis.set_ylabel('Derivative')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_actions(self, axis):
        """Plot the current (blue) and next action (red) over time in
        ``axis``."""
        a_curr = self.stack_data('a_curr')
        a_next = self.stack_data('a_next')
        axis.plot(a_curr, 'b', label='a_curr')
        axis.plot(a_next, 'r', label='a_next')
        axis.legend(loc=0)
        axis.set_xlabel('step')
        axis.set_ylabel('Action')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_error(self, axis):
        """Plot the error over time in ``axis``."""
        data = self.stack_data('err')
        axis.plot(data, 'k', label='error')
        axis.set_xlabel('step')
        axis.set_ylabel('TD-Error')
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_accumulated_reward(self, axis):
        """Plot the accumulated reward per episode in ``axis``."""
        reward = self.get_data('reward')
        data = [r.sum() for r in reward]
        axis.plot(data, 'k', label='Accumulated reward')
        axis.set_xlabel('episode')
        axis.set_ylabel('Accumulated reward')
        return axis
    
    def plot_return_prediction(self, axis):
        """Plot the predicted return of the current (red) and next
        (blue) state/action pair in ``axis``."""
        j_curr = self.stack_data('j_curr')
        j_next = self.stack_data('j_next')
        axis.plot(j_curr, 'r', label='j_curr')
        axis.plot(j_next, 'b', label='j_next')
        axis.set_xlabel('step')
        axis.set_ylabel('predicted return')
        axis.legend(loc=0)
        if self.always_plot_grid:
            self.plot_grid(ax)
        return axis
    
    def plot_path_return_prediction(self, axis, expno):
        """Plot the predicted return of a simulated path.
        
        The return prediction is specific to the path of the experiment
        *expno*. The plots will be written to *axis*.
        
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
        
        trg = np.zeros((N, 1))
        for i in range(N-2, -1, -1):
            trg[i] = trg[i+1] * gamma[i+1] + reward[i+1]
        
        trg[-1] = trg[-2]
        
        import mdp
        from rc import StabilizedRLS
        
        
        lin_reg = mdp.nodes.LinearRegressionNode(input_dim=reservoir_dim, output_dim=1)
        lin_reg.train(x_curr[:-1], trg[:-1])
        lin_reg.stop_training()
        
        olr_nb = StabilizedRLS(with_bias=False, input_dim=reservoir_dim, output_dim=1, lambda_=0.9)
        olr_wb = StabilizedRLS(with_bias=True,  input_dim=reservoir_dim, output_dim=1, lambda_=0.9)
        for i in range(N-1):
            src = np.atleast_2d(x_curr[i])
            dst = np.atleast_2d(trg[i])
            olr_wb.train(src, dst)
            olr_nb.train(src, dst)
        
        axis.plot(lin_reg(x_curr), 'b', label='Offline MSE')
        axis.plot(olr_nb(x_curr), 'c--', label='Online MSE (no bias)')
        axis.plot(olr_wb(x_curr), 'c:', label='Online MSE (bias)')
        axis.plot(j_curr, 'r', label='Predicted Value')
        axis.plot(trg, 'k', label='target')
        axis.legend(loc=0)
        axis.set_xlabel('step')
        axis.set_ylabel('return')
        return axis
    
    def plot_predicted_return_over_episodes(self, axis, step=1):
        """Plot the evolution of the predicted return over multiple
        episodes.
        
        This function assumes that the same path (same action sequence)
        was applied in all experiments.
        
        """
        j_curr = self.get_data('j_curr')
        reward = self.get_data('reward')[0]
        gamma = self.get_data('gamma')[0]
        
        N = reward.shape[0]
        trg = np.zeros((N, 1))
        for i in range(N-2, -1, -1):
            trg[i] = trg[i+1] * gamma[i+1] + reward[i+1]
        
        trg[-1] = trg[-2]
        axis.plot(trg, 'k', label='Return', linewidth='2')

        for lbl, j_per_episode in zip(self.experiments[::step], j_curr[::step]):
            axis.plot(j_per_episode, label='Episode ' + lbl)
        
        axis.legend(loc=0)
        axis.set_xlabel('step')
        axis.set_ylabel('Predicted return')
        return axis
    
    def plot_error_over_episodes(self, axis, step=1):
        """Plot the evolution of the error over multiple episodes.
        
        This function assumes that the same path (same action sequence)
        was applied in all experiments.
        
        """
        err = self.get_data('err')
        for lbl, j_per_episode in zip(self.experiments[::step], err[::step]):
            axis.plot(j_per_episode, label='Episode ' + lbl)
        
        axis.legend(loc=0)
        axis.set_xlabel('step')
        axis.set_ylabel('TD-Error')
        return axis
    
    def plot_reservoir_input(self, axis):
        """Plot the reservoir input over time in ``axis``. Per input,
        one line is drawn."""
        data = self.stack_data('i_curr')
        N,M = data.shape
        for i in range(M):
            axis.plot(data[:,i], label='Input %i'%i)
        
        axis.set_xlabel('step')
        axis.set_ylabel('Input')
        axis.legend(loc=0)
        return axis
    
    def plot_input_over_episode(self, axis, episode):
        data = self.f[episode]['i_curr'][:]
        N,M = data.shape
        for i in range(M):
            axis.plot(data[:,i], label='Input %i'%i)
        
        axis.set_xlabel('step')
        axis.set_ylabel('Input')
        axis.legend(loc=0)
        return axis
    
    def plot_node_over_episode(self, axis, episode, node):
        """Plot the reservoir input and output of ``node`` on a tanh
        shape over one ``episode`` in ``axis``. If ``node`` is None,
        all nodes are plotted. If ``axis`` is a list, each node is
        plotted in its own axis."""
        
        assert episode in self.experiments
        
        o_output = self.f[episode]['x_curr'][:,node]
        o_input  = np.arctanh(o_output)
        
        if node is None:
            raise NotImplementedError()
        
        # plot tanh
        lo = min(-5, o_input.min())
        hi = max(5, o_input.max()) 
        range_x = np.arange(lo, hi, 0.1)
        axis.plot(range_x, np.tanh(range_x), label='tanh', color='0.75')
        
        # plot points
        axis.plot(o_input, o_output, 'b*', label='samples')
        axis.set_xlabel('node input')
        axis.set_ylabel('node output')
        return axis
    
    def plot_node_over_episode_time_input(self, axis, episode, node):
        """Plot the time over input of ``node`` in ``axis``. Only
        ``episode`` is considered."""
        assert episode in self.experiments
        
        o_output = self.f[episode]['x_curr'][:,node]
        o_input  = np.arctanh(o_output)
        
        if node is None:
            raise NotImplementedError()
        
        #o_time = range(o_input.shape[0]-1, -1, -1)
        #yticks = range(o_input.shape[0])
        #axis.set_yticks(yticks)
        o_time = range(o_input.shape[0])
        axis.plot(o_input, o_time, 'k')
        axis.set_xlabel('node input')
        axis.set_ylabel('time')
        return axis
    
    def plot_node_over_episode_time_output(self, axis, episode, node):
        """Plot the output of ``node`` over time in ``axis``. Only
        ``episode`` is considered."""
        assert episode in self.experiments
        
        o_output = self.f[episode]['x_curr'][:,node]
        o_input  = np.arctanh(o_output)
        
        if node is None:
            raise NotImplementedError()
        
        o_time = range(o_output.shape[0])
        axis.plot(o_time, o_output, 'k')
        axis.set_xlabel('time')
        axis.set_ylabel('node output')
        return axis


def overview(analysis, figure):
    """Plot some characteristics of ``analysis`` in ``figure``."""
    analysis.plot_readout(figure.add_subplot(321))
    analysis.plot_reward(figure.add_subplot(322))
    analysis.plot_derivative(figure.add_subplot(323))
    analysis.plot_actions(figure.add_subplot(324))
    analysis.plot_error(figure.add_subplot(325))
    analysis.plot_accumulated_reward(figure.add_subplot(326))
    figure.suptitle('Some overview characteristics')
    return figure

def node_inspection(analysis, figure, episode, node):
    """Plot input/output of reservoir node ``node`` at ``episode``
    according to ``analysis`` in ``figure``."""
    
    # plots
    ax_main = analysis.plot_node_over_episode(figure.add_subplot(221), episode, node)
    ax_out = analysis.plot_node_over_episode_time_output(figure.add_subplot(222), episode, node)
    ax_in = analysis.plot_node_over_episode_time_input(figure.add_subplot(223), episode, node)
    ax_ep_input = analysis.plot_input_over_episode(figure.add_subplot(224), episode)
    
    # equal axis 
    [xmin, xmax, ymin, ymax] = ax_main.axis()
    
    ax_in.axis(xmin=xmin, xmax=xmax) # keep y axis
    ax_out.axis(ymin=ymin, ymax=ymax) # keep x axis
    
    figure.suptitle('Characteristics of node %i at episode %s' % (node, episode))
    return figure
