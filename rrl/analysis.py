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

Plots from papers
* reward over time; one line per episode, received and predicted
* utility distribution estimated by critic at a specific time step
* change of w_out over time
* utility and prediction (and raw sensor values) over time
* measured and predicted reward over time, one line per episode

If the controller is working, we should see that
* If training is executed on always the same sequence of actions,
  the value prediction should become more accurate for earlier
  steps with increasing number of experiments.
  > This is since for estimating the TD-Error, we're relying on the current
    value to improve the estimate of the previous one
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
