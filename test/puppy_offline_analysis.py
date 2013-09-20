import rrl
import pylab
import numpy as np
import h5py
import sys

# global config var
step_width = 150
step_width_plotting = 50

# the observations file lists the initial epoch, while the critic datafile doesn't
# thus, the sensor data must be shifted by one step_width
obs_offset = step_width
# The analysis experiments are always reverted, hence there's only one initial sample
# (check out puppy_offline_playback:"if 'init_step' in data_grp: [...]", ActorCritic.__call__ and PuppyHDP.init_episode)
# For this initial sample, nothing is written into the analysis_critic_pth file. Hence, there's an offset
# of one epoch for data in analysis_data_pth and analysis_critic_pth.
# Note that if the experiments are restarted instead of reverted, this offset would be =2

robot_radius = 0.2


# Open files
a = rrl.Analysis(rrl.DataMerge('/tmp/example_eval.hdf5', '/tmp/example_data.hdf5'))

# Create figure
fig = pylab.figure()
axis = fig.add_subplot(111)

# Plot target
rrl.puppy_plot_locationtarget(axis, target=target_loc, distance=0.5)
axis.invert_xaxis() # positive x-axis in webots goes to the left!
pylab.show(block=False)

# Retrieve and plot the initial trajectory
grp = a['0'] # this is assumed to be the main trajectory
main_pth = grp['a_curr'][:]
main_len = main_pth.shape[0] * step_width
rrl.puppy_plot_trajectory(a, axis, '0', step_width, offset=step_width*25, label='Initial trajectory')
pylab.show(block=False)

def find_offset(a0, a1):
    """
    """
    offset = min(a0.shape[0], a1.shape[0])
    while not (a0[:offset] == a1[:offset]).all():
        offset -= 1
        if offset < 0:
            raise IndexError()
    
    return offset

# group experiments with respect to the main trajectory cutoff and also
# get normalization data
pth_data = {}
for expno in a.experiments:
    if expno == '0':
        # '0' is the vanilla trajectory, don't consider it
        continue
    
    grp = a[expno]
    data_offset = find_offset(main_pth, grp['a_curr'][:])
    
    if data_offset not in pth_data:
        pth_data[data_offset] = []
    
    pth_data[data_offset].append((expno, grp['j_curr'][-3]))

# Compute normalization params over the whole experiment
returns_total = np.vstack([map(lambda i: i[1], lst) for lst in pth_data.values()])
nrm_total_min = returns_total.min()
nrm_total_ptp = returns_total.ptp()

# Go through data, plot the actions/returns
for data_offset in pth_data:
    
    # get data
    experiments, nrm_data = zip(*pth_data[data_offset])
    
    # Compute the normalization params over the current state
    p_returns = np.hstack(nrm_data)
    p_min = p_returns.min()
    p_ptp = p_returns.ptp()
    
    # Plot the robot disc
    if len(pth_data[data_offset]) > 1:
        loc_robot = (a['0']['puppyGPS_x'][(data_offset+1)*step_width-1], a['0']['puppyGPS_y'][(data_offset+1)*step_width-1])
        robot_color = (np.median(p_returns) - nrm_total_min) / (nrm_total_ptp)
        robot_color = 1.0 - robot_color
        rob = pylab.Circle(loc_robot, robot_radius, fill=True, facecolor=str(robot_color))
        axis.add_artist(rob)
    
    # Plot the rays
    for expno, return_ in pth_data[data_offset]:
        grp = a[expno]
        lbl = expno
        
        sensor_offset = obs_offset + data_offset * step_width
        data_x_plot = grp['puppyGPS_x'][sensor_offset-1::step_width_plotting]
        data_y_plot = grp['puppyGPS_y'][sensor_offset-1::step_width_plotting]

        col = 0.25 + (return_ - p_min) / (2.0 * p_ptp+1e-7)
        col = 1.0 - col
        col = col[0]
        
        axis.plot(data_x_plot, data_y_plot, linewidth=2, label=lbl, color=str(col))
        pylab.draw()
