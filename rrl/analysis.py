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

"""
vl = np.array(steps).cumsum()[_max_episodes-_test_episodes]
def grid():
    for i in np.array(steps).cumsum():
        pylab.axvline(i, linestyle='-', color='0.7')

# Plot the absolute readout weights
pylab.subplot(321)
pylab.plot(abs(query('readout')).sum(axis=1), 'k', label='readout weights')
pylab.title('Sum of absolute readout weights')
#pylab.axvline(vl)
#pylab.yscale('log')
grid()

# Plot the reward
pylab.subplot(322)
pylab.plot(query('reward'), 'k', label='reward')
pylab.title('Reward per step')
#pylab.axvline(vl)
grid()

# Plot the Derivation
pylab.subplot(323)
pylab.plot(query('deriv'), 'k', label='Derivative')
pylab.title('Derivative')
#pylab.axvline(vl)
#pylab.yscale('log')
grid()

# Plot the actions
pylab.subplot(324)
pylab.plot(query('a_curr'), 'b', label='a_curr')
pylab.plot(query('a_next'), 'r', label='a_next')
pylab.axhline(2*pi, color='k')
pylab.axhline(0, color='k')
pylab.title('Action')
#pylab.axvline(vl)
pylab.legend(loc=0)
grid()

# Plot error
pylab.subplot(325)
pylab.plot(query('err'), 'k', label='error')
pylab.title('TD-Error')
#pylab.axvline(vl)
#pylab.yscale('log')
grid()

# Accumulated reward per episode
pylab.subplot(326)
#pylab.plot(steps, 'k')
pylab.plot([h['reward'].sum() for h in history])
pylab.title('Accumulated reward per episode')
#pylab.axvline(_max_episodes-_test_episodes)
#grid()

pylab.subplots_adjust(
    left=0.07,
    bottom=0.05,
    right=0.96,
    top=0.94,
    wspace=0.21,
    hspace=0.29
)

pylab.show(block=False)



# Offline reservoir training:
pylab.figure(4)
# Setup needs to be: large number of experiments, always the same one (simulation), wall experiment
hidx,cidx=-1,-1
src = history[hidx]['x_curr']
N = src.shape[0]
trg = np.zeros((N,1))
for i in range(N-2,-1,-1):
    trg[i] = trg[i+1] * gamma + history[hidx]['reward'][i+1]

trg[-1] = trg[-2] # np.float('nan')


import mdp
lr = mdp.nodes.LinearRegressionNode(input_dim=reservoir_dim,output_dim=1)
lr.train(src[:-1],trg[:-1])

olrA = PlainRLS(with_bias=False, input_dim=reservoir_dim, output_dim=1, lambda_=1.0)
olrB = PlainRLS(with_bias=True, input_dim=reservoir_dim, output_dim=1, lambda_=1.0)
for i in range(src.shape[0]-1):
    s = np.atleast_2d(src[i])
    t = np.atleast_2d(trg[i])
    olrA.train(s,t)
    olrB.train(s,t)

pylab.title('Next-step return prediction of some methods (training case)')
#pylab.plot(history[hidx]['reward'], 'y', label='Reward')
pylab.plot(lr(src), 'b', label='Offline MSE')
pylab.plot(olrA(src), 'c--', label='Online MSE (no bias)')
pylab.plot(olrB(src), 'c:', label='Online MSE (bias)')
pylab.plot(history[cidx]['j_curr'], 'r', label='Online TD')
pylab.plot(trg, 'k', label='target')
pylab.legend(loc=0)
pylab.xlabel('step')
pylab.ylabel('return')
pylab.show(block=False)
"""
