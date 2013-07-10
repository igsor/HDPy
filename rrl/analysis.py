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

"""
