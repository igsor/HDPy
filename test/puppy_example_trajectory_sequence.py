
import h5py
import numpy as np

# Number of steps that will always be executed
init_steps = 4
# Sample action execution step increment
step_size = 3
# Target location
sequence_file = '/tmp/example_sequence.hdf5'

# main trajectory
main_trajectory = [[0.8, 0.8]] * 3 + [[0.8, 0.78]] * 2 + [[0.82, 0.9]] * 10 + [[0.8, 0.9]] * 5 + [[0.82, 0.9]] * 10
main_trajectory = np.array(main_trajectory)

# example actions, to be executed at some steps of the main trajectory
action_samples = np.array([
    [ 0.4,  0.8],
    [ 0.8,  0.4],
    [ 0.6,  0.8],
    [ 0.8,  0.6],
    [ 0.8,  1. ],
    [ 1. ,  0.8],
    [ 0.8,  0.8]
])

# Create the trajectories to be executed. This is the main trajectory
# up to step i, then each example action for three steps
ex_trajectory = [main_trajectory]
for i in range(init_steps, main_trajectory.shape[0] + 1, step_size):
    ex_trajectory += [np.vstack((main_trajectory[:i], sample, sample, sample)) for sample in action_samples]

# Store the example trajectories in a HDF5 file
# The trajectories are stored in seperate datasets (traj_000) and an
# index (idx) is initialized for progress bookkeeping.
f = h5py.File(sequence_file, 'w')
f.create_dataset('idx', data=0)
f.create_dataset('main', data=main_trajectory)
for idx, traj in enumerate(ex_trajectory):
    name='traj_%03i' % idx
    f.create_dataset(name, data=traj)

f.close()

print "Stored", len(ex_trajectory), "sequences"
