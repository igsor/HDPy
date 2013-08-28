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

* Reservoir inputs; Goes together with reservoir output (neuron potential)

  > ``plot_reservoir_input``, ``plot_node_over_episode``

* The ratio between the input and previous state of one reservoir node.

* Snapshot (A):

  * For some sample trajectories:
  
    - At each step, evaluate the Critic dependent on the action (finite, predefined set)
    - For each evaluated action, plot a ray. Expected return is encoded as color or ray length
    - Plot the action chosen by the algorithm (for a policy study)
    
  * At each point, the critic is evaluated dependent on the action
  * The critic is somewhat also evaluated dependent on the state, since we have a trajectory.
  * If the critic was properly trained, we should see that:
  
    - The predicted return may be "fuzzy" in the beginning (a lot of possible trajectories and returns)
    - The predicted return should be more accurate (w.r.t. reward) for close-to final episode steps (few possible trajectories)
    - The predicted return should somehow reflect the naturally expected return; Close to the wall gives a smaller return than far away
    - The chosen action should be close to the best (evaluated) action


* Snapshot (B):

  * For some sample trajectories:
  
    - The action remains constant along the trajectory
    - Evaluate the return at the last step (possibly dependent on the action)

? Correctness of reward

? Predicted return

? Maybe an 'animation' (characteristics over time) may be informative


"""

import numpy as np
import h5py
from math import pi
import inout
import warnings

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
    def __init__(self, pth, grid=False, min_key_length=0):
        if isinstance(pth, inout.DataMerge):
            self.pth = pth.pth0
            self.f = pth
        else:
            self.pth = pth
            self.f = h5py.File(pth, 'r')
        exp = map(int, self.f.keys())
        exp = sorted(exp)
        exp = map(str, exp)
        exp = filter(lambda k: len(self.f[k]) > min_key_length, exp) # normalization fails, if not >0
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
    
    def __getitem__(self, key):
        if isinstance(key, int):
            warnings.warn("Warning: access by int is w.r.t the experiment not the group")
            key = self.experiments[key]
        
        return self.get_episode(key)
    
    def __len__(self):
        return len(self.experiments)
    
    def get_episode(self, num):
        assert num in self.experiments
        return self.f[num]
    
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
    
    def episode_len(self):
        return [len(self.f[e]['reward']) for e in self.experiments]
    
    def plot_episode_len(self, axis, **kwargs):
        """Plot the length of the episodes in ``axis``."""
        data = self.episode_len()
        color = kwargs.pop('color', 'k')
        axis.plot(data, color=color, **kwargs)
        axis.set_xlabel('episode')
        axis.set_ylabel('number of steps')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
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
    
    def plot_grid(self, axis, key=None):
        """Add a vertical bar to ``axis`` to mark experiment
        boundaries."""
        steps = self.num_steps(key)
        for i in np.array(steps).cumsum():
            axis.axvline(i, linestyle='-', color='0.7')
        
        return axis
    
    def plot_neuron_activation(self, axis):
        """Plot the absolute sum of the neuron state (=activation
        potential) of the current (red) and next (blue) timestep in
        ``axis``."""
        x_curr = self.stack_data('x_curr')
        x_next = self.stack_data('x_next')
        N = x_curr.shape[1]
        axis.plot(abs(x_curr).sum(axis=1)/N, 'r', label='Absolute Neuron Activation')
        axis.plot(abs(x_next).sum(axis=1)/N, 'b', label='Absolute Neuron Activation')
        axis.set_xlabel('step')
        axis.set_ylabel('Absolute Neuron Activation')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_readout(self, axis, func=lambda i:i):
        """Plot the readout nodes as individual curves in ``axis``. The
        ``func`` allows for an operation to applied to the data before
        plotting. The main intention for this is :py:func:`abs`. Default
        is the identity."""
        data = self.stack_data('readout')
        N = data.shape[1]
        for i in range(N):
            if i == N-1:
                lbl = 'Bias'
            else:
                lbl = 'Readout %i' % i
            
            axis.plot(func(data[:, i]), label=lbl)
        
        #axis.legend(loc=0)
        axis.set_xlabel('step')
        axis.set_ylabel('readout weight')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_readout_sum(self, axis, **kwargs):
        """Plot the absolute readout weight over time in ``axis``."""
        data = self.stack_data('readout')
        color = kwargs.pop('color', 'k')
        axis.plot(abs(data).sum(axis=1), color=color, **kwargs)
        axis.set_xlabel('step')
        axis.set_ylabel('Sum of absolute readout weights')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_readout_diff(self, axis, **kwargs):
        """Plot the difference of absolute readout weights in
        ``axis``."""
        data = self.stack_data('readout')
        diff = data[1:] - data[:-1]
        diff = diff.sum(axis=1)**2
        color = kwargs.pop('color','k')
        axis.plot(diff, color=color, label='Readout difference', **kwargs)
        axis.set_xlabel('step')
        axis.set_ylabel('Readout difference')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_cumulative_readout_diff(self, axis, **kwargs):
        """Plot the cumulative difference of absolute readout weights in
        ``axis``."""
        data = self.stack_data('readout')
        diff = data[1:] - data[:-1]
        diff = diff.sum(axis=1)**2
        roc = diff.cumsum() / diff.sum()
        
        color = kwargs.pop('color', 'k')
        axis.plot(roc, color=color, **kwargs)
        axis.set_ylabel('Cumulative readout weight distance')
        axis.set_xlabel('step')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_node_weight_over_episode(self, axis, episode):
        """Plot the readout weights of a single ``episode`` in ``axis``."""
        assert episode in self.experiments
        data = self.f[episode]['readout'][:]
        N = data.shape[1]
        for i in range(N):
            if i == N-1:
                lbl = 'Bias'
            else:
                lbl = 'Node %i' % i
            axis.plot(data[:, i], label=lbl)
        
        axis.set_xlabel('step')
        axis.set_ylabel('Node weight')
        axis.legend(loc=0)
        return axis
    
    def plot_reward(self, axis, **kwargs):
        """Plot the reward over time in ``axis``."""
        data = self.stack_data('reward')
        color = kwargs.pop('color', 'k')
        axis.plot(data, color=color, **kwargs)
        axis.set_xlabel('step')
        axis.set_ylabel('Reward')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_derivative(self, axis):
        """Plot the derivative dJ/da over time in ``axis``."""
        data = self.stack_data('deriv')
        axis.plot(data, 'k', label='dJ(k)/da')
        axis.set_xlabel('step')
        axis.set_ylabel('Derivative')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_actions(self, axis):
        """Plot the current (blue) and next action (red) over time in
        ``axis``."""
        a_curr = self.stack_data('a_curr')
        a_next = self.stack_data('a_next')
        axis.plot(a_curr, 'b', label='a_curr')
        axis.plot(a_next, 'r', label='a_next')
        #axis.legend(loc=0)
        axis.set_xlabel('step')
        axis.set_ylabel('Action')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_absolute_error(self, axis):
        """Plot the absolute error over time in ``axis``."""
        data = abs(self.stack_data('err'))
        axis.plot(data, 'k', label='error')
        axis.set_xlabel('step')
        axis.set_ylabel('TD-error')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_error(self, axis):
        """Plot the error over time in ``axis``."""
        data = self.stack_data('err')
        axis.plot(data, 'k', label='error')
        axis.set_xlabel('step')
        axis.set_ylabel('TD-error')
        if self.always_plot_grid:
            self.plot_grid(axis)
        return axis
    
    def plot_accumulated_reward(self, axis, **kwargs):
        """Plot the accumulated reward per episode in ``axis``."""
        reward = self.get_data('reward')
        data = [r.sum() for r in reward]
        color = kwargs.pop('color', 'k')
        axis.plot(data, color=color, **kwargs)
        #axis.set_xticks(self.experiments)
        axis.set_xlabel('episode')
        axis.set_ylabel('Accumulated reward')
        return axis
    
    def plot_mean_reward(self, axis):
        """Plot the accumulated reward per episode in ``axis``."""
        reward = self.get_data('reward')
        data = [r.mean() for r in reward]
        axis.plot(data, 'k', label='Accumulated reward')
        #axis.set_xticks(self.experiments)
        axis.set_xlabel('episode')
        axis.set_ylabel('Accumulated reward')
        return axis
    
    def plot_return_prediction(self, axis, plot=('j_curr', 'j_next')):
        """Plot the predicted return of the current (red) and next
        (blue) state/action pair in ``axis``. ``plot`` defines which
        of the two predicted return are plotted."""
        if 'j_curr' in plot:
            j_curr = self.stack_data('j_curr')
            axis.plot(j_curr, 'r', label='j_curr')
        
        if 'j_next' in plot:
            j_next = self.stack_data('j_next')
            axis.plot(j_next, 'b', label='j_next')
        
        axis.set_xlabel('step')
        axis.set_ylabel('predicted return')
        if type(plot) != type('') and len(plot) > 1:
            axis.legend(loc=0)
        
        if self.always_plot_grid:
            self.plot_grid(axis)
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
        
        #axis.plot(lin_reg(x_curr), 'b', label='Offline MSE')
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
        N = data.shape[1]
        for i in range(N):
            axis.plot(data[:, i], label='Input %i' % i)
        
        axis.set_xlabel('step')
        axis.set_ylabel('Input')
        axis.legend(loc=0)
        return axis
    
    def plot_input(self, axis):
        """Plot the reservoir input in ``axis``."""
        data = self.stack_data('i_curr')
        axis.plot(data, label='Input')
        axis.set_xlabel('step')
        axis.set_ylabel('Input')
        return axis
    
    def plot_input_over_episode(self, axis, episode):
        """Plot the reservoir input of a single ``episode`` in ``axis``."""
        data = self.f[episode]['i_curr'][:]
        N = data.shape[1]
        for i in range(N):
            axis.plot(data[:, i], label='Input %i' % i)
        
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
        
        o_output = self.f[episode]['x_curr'][:, node]
        o_input  = np.arctanh(o_output)
        
        if node is None:
            raise NotImplementedError()
        
        # plot tanh
        o_input_noninf = o_input[np.isfinite(o_input)]
        r_min = min(-5, o_input_noninf.min())
        r_max = max(5, o_input_noninf.max()) 
        range_x = np.arange(r_min, r_max, 0.1)
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
        
        o_output = self.f[episode]['x_curr'][:, node]
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
        
        o_output = self.f[episode]['x_curr'][:, node]
        
        if node is None:
            raise NotImplementedError()
        
        o_time = range(o_output.shape[0])
        axis.plot(o_time, o_output, 'k')
        axis.set_xlabel('time')
        axis.set_ylabel('node output')
        return axis

def overview(analysis, figure):
    """Plot some characteristics of ``analysis`` in ``figure``."""
    analysis.plot_readout_sum(figure.add_subplot(321))
    analysis.plot_reward(figure.add_subplot(322))
    analysis.plot_derivative(figure.add_subplot(323))
    analysis.plot_actions(figure.add_subplot(324))
    analysis.plot_error(figure.add_subplot(325))
    analysis.plot_accumulated_reward(figure.add_subplot(326))
    #figure.suptitle('Some overview characteristics')
    return figure

def node_inspection(analysis, figure, episode, node):
    """Plot input/output of reservoir node ``node`` at ``episode``
    according to ``analysis`` in ``figure``."""
    
    # plots
    ax_main = analysis.plot_node_over_episode(figure.add_subplot(221), episode, node)
    ax_out = analysis.plot_node_over_episode_time_output(figure.add_subplot(222), episode, node)
    ax_in = analysis.plot_node_over_episode_time_input(figure.add_subplot(223), episode, node)
    analysis.plot_input_over_episode(figure.add_subplot(224), episode)
    
    # equal axis 
    [xmin, xmax, ymin, ymax] = ax_main.axis()
    
    ax_in.axis(xmin=xmin, xmax=xmax) # keep y axis
    ax_out.axis(ymin=ymin, ymax=ymax) # keep x axis
    
    figure.suptitle('Characteristics of node %i at episode %s' % (node, episode))
    return figure

def critic(plant, reservoir, readout, norm=None):
    """Use the simulation parts to set up a simpler to use critic.
    The critic is a function of the *state* and *action*. Furthermore,
    it takes *simulate* as argument to control if the reservoir state
    is actually advanced."""
    if norm is None:
        import PuPy
        norm = PuPy.Normalization()
    else:
        plant.set_normalization(norm)
    
    if 'a_curr' not in norm:
        norm.set('a_curr', 0.0, 1.0)
    
    def critic_fu(state, action, simulate):
        """Return the expected return according to the *critic* with
        input ``state`` and ``action``. If
        ``simulate`` = :py:keyword:`True`, the output is computed but
        the reservoir not updated."""
        in_state = plant.state_input(state)
        action_nrm = norm.normalize_value('a_curr', action)
        i_curr = np.vstack((in_state, action_nrm)).T
        x_curr = reservoir(i_curr, simulate=simulate)
        #o_curr = np.hstack((x_curr, i_curr)) # FIXME: Input/Output ESN Model
        #j_curr = readout(o_curr) # FIXME: Input/Output ESN Model
        j_curr = readout(x_curr)
        return j_curr
    
    return critic_fu
