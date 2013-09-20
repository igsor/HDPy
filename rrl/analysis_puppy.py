"""
Analysis tools, specific for the *Puppy* experiment.
"""

import pylab
import numpy as np
import itertools
from puppy import SENSOR_NAMES

def puppy_plot_trajectory(analysis, axis, episode, step_width=1, offset=0, legend=True, **kwargs):
    """Plot the trajectory of an episode
    """
    gps_x = analysis[episode]['puppyGPS_x'][offset+step_width-1::step_width]
    gps_y = analysis[episode]['puppyGPS_y'][offset+step_width-1::step_width]
    if step_width > 1:
        gps_x = np.concatenate(([analysis[episode]['puppyGPS_x'][offset]], gps_x))
        gps_y = np.concatenate(([analysis[episode]['puppyGPS_y'][offset]], gps_y))

    col = kwargs.pop('color', 'k')
    label = kwargs.pop('label', 'Trajectory')
    axis.plot(gps_x, gps_y, color=col, linewidth=3, **kwargs)
    axis.axis('equal')
    if legend:
        axis.plot(gps_x[0], gps_y[0], 'ks', label='Start')
        axis.plot(gps_x[-1], gps_y[-1], 'kv', label='End')
    
    return axis

def puppy_plot_all_trajectories(analysis, axis, step_width=1, **kwargs):
    """Plot all trajectories in ``analysis`` into ``axis``.
    """
    gps_x = analysis.get_data('puppyGPS_x')
    gps_y = analysis.get_data('puppyGPS_y')
    
    N = len(gps_x)-1
    kwargs.pop('color', None) # remove color argument
    for idx, (x, y) in enumerate(zip(gps_x, gps_y)):
        col = 0.75 - (0.75 * (idx - 1))/N
        
        x_plot = np.concatenate(([x[0]], x[step_width-1::step_width]))
        y_plot = np.concatenate(([y[0]], y[step_width-1::step_width]))
        
        axis.plot(x_plot, y_plot, color=str(col), **kwargs)
    
    return axis

def puppy_plot_linetarget(axis, origin=(2.0, 0.0), direction=(1.0, 1.0), range_=(-5.0, 5.0)):
    """Plot a line given by ``origin`` and ``direction``. The ``range_``
    may be supplid, which corresponds to the length of the line (from
    the origin).
    """
    origin = np.array(origin)
    dir_ = np.array(direction)
    dir_ /= np.linalg.norm(dir_)
    line = [origin + t * dir_ for t in range_]
    line_x, line_y = zip(*line)
    axis.plot(line_x, line_y, 'k', label='Target')
    return axis

def puppy_plot_locationtarget(axis, target=(4.0, 4.0), distance=0.5, **kwargs):
    """Plot the ``target`` location with a sphere of radius ``distance``
    into ``axis`` to mark the target location. ``kwargs`` will be passed
    to all :py:mod:`pylab` calls."""
    linewidth = kwargs.pop('linewidth', 2)
    color = kwargs.pop('facecolor', 'k')
    fill = kwargs.pop('fill', False)
    lbl = kwargs.pop('label', 'Target')
    axis.plot([target[0]], [target[1]], 'kD', label=lbl, **kwargs)
    if distance > 0.0:
        trg_field = pylab.Circle(target, distance, fill=fill, facecolor=color, linewidth=linewidth, label=lbl, **kwargs)
        axis.add_artist(trg_field)

    return axis

def puppy_plot_landmarks(axis, landmarks, **kwargs):
    """Plot markers at ``landmark`` locations in ``axis``."""
    color = kwargs.pop('color', 'k')
    lbl = kwargs.pop('label', '')
    marker = kwargs.pop('marker','^')
    for x,y in landmarks:
        axis.plot([x],[y], marker=marker, color=color, label=lbl, **kwargs)
    return axis

def puppy_plot_action(analysis, episode, critic, reservoir, inspect_epochs, actions_range_x, actions_range_y, step_width, obs_offset, epoch_actions=None):
    """
    
    .. todo::
        documentation, video mode
    
    """
    grp = analysis[episode]
    if epoch_actions is None:
        epoch_actions = [None] * len(inspect_epochs)
    
    for trg_epoch, actions in zip(inspect_epochs, epoch_actions):
        
        # simulate the actions
        a_ret = _puppy_action_eval(grp, reservoir, critic, trg_epoch, obs_offset, step_width, actions_range_x, actions_range_y)
        
        # plot results
        fig = pylab.figure()
        # In the image, the y-axis is the rows, the x-axis the columns of the matrix
        # Having index (0,0) in the left/bottom corner: origin='lower'
        pylab.plot((0, len(actions_range_x)-1), (0, len(actions_range_y)-1), 'b')
        pylab.imshow(a_ret, origin='lower', cmap=pylab.cm.gray)
        pylab.colorbar()
        pylab.xticks(range(len(actions_range_y)), actions_range_y)
        pylab.yticks(range(len(actions_range_x)), actions_range_x)
        pylab.title('Expected Return per action at epoch ' + str(trg_epoch))
        pylab.xlabel('Amplitude right legs') # cols are idx_y, right legs
        pylab.ylabel('Amplitude left legs') # rows are idx_x, left legs
    
        if actions is not None:
            a_left, a_right = zip(*actions)
            pylab.plot(a_left, a_right, 'r')
            pylab.plot([a_left[0]], [a_right[0]], 'rs')
    
    return fig

def puppy_plot_inspected_trajectory(analysis, episode_idx, step_width, axis, inspect_epochs, obs_offset):
    puppy_plot_trajectory(analysis, axis, episode_idx, step_width, color='b', offset=obs_offset)
    trg_x = [analysis[episode_idx]['puppyGPS_x'][obs_offset + step_width*trg_epoch+step_width-1] for trg_epoch in inspect_epochs]
    trg_y = [analysis[episode_idx]['puppyGPS_y'][obs_offset + step_width*trg_epoch+step_width-1] for trg_epoch in inspect_epochs]
    axis.plot(trg_x, trg_y, 'k*', label='Inspected states')
    return axis

def _puppy_action_eval(grp, reservoir, critic, trg_epoch, obs_offset, step_width, actions_range_x, actions_range_y):
    reservoir.reset()
    #reservoir.states = np.atleast_2d(grp['x_next'][trg_epoch-1-25,:])
    #reservoir.states = np.atleast_2d(grp['x_curr'][trg_epoch-1,:])
    reservoir.states = np.atleast_2d(grp['x_curr'][trg_epoch-1,:reservoir.get_output_dim()])
    
    # evaluate actions
    # Note: epoch is one step ahead (of a_curr, same time as a_next)!
    # Note: sensor values are shifted w.r.t a_curr by obs_offset
    s_curr = dict([(sensor,grp[sensor][obs_offset+step_width*(trg_epoch-1):obs_offset+trg_epoch*step_width]) for sensor in SENSOR_NAMES])
    a_ret = np.zeros((len(actions_range_x), len(actions_range_y)))
    
    actions_iter = itertools.product(range(len(actions_range_x)), range(len(actions_range_y)))
    for idx_x, idx_y in actions_iter:
        action_candidate = np.atleast_2d((actions_range_x[idx_x], actions_range_y[idx_y])).T
        j_curr = critic(s_curr, action_candidate, simulate=True)
        a_ret[idx_x, idx_y] = j_curr[0, 0]
        #print actions_range_x[idx_x], actions_range_y[idx_y], j_curr[0, 0]
    
    return a_ret

def puppy_vid_action(image, (a_line, a_marker, px, py), grp, critic, reservoir, epoch, actions_range_x, actions_range_y, step_width, obs_offset, actions=None, actions_keep=False):
    """
    
    .. todo::
        documentation,
    
    """
    a_ret = _puppy_action_eval(grp, reservoir, critic, epoch, obs_offset, step_width, actions_range_x, actions_range_y)
    
    # update plot
    image.set_data(a_ret)
    image.set_clim(vmin=a_ret.min(), vmax=a_ret.max())
    image.changed()
    
    if actions is not None:
        actions[:,0] = (actions[:,0] - py[0]) * py[1]
        actions[:,1] = (actions[:,1] - px[0]) * px[1]
        a_line.set_data((actions[:,1], actions[:,0]))
        a_marker.set_data(([actions[-1,1]], [actions[-1,0]]))
    
    return image

def puppy_vid_init(actions_range_x, actions_range_y, with_actions=True):
    """
    
    .. todo::
        The action isn't displayed correctly (offset?)
    
    """
    fig = pylab.figure()
    axis = fig.add_subplot(111)
    axis.set_xticks(range(len(actions_range_y)))
    axis.set_xticklabels(actions_range_y)
    axis.set_yticks(range(len(actions_range_x)))
    axis.set_yticklabels(actions_range_x)
    #title = fig.suptitle('Expected Return per action')
    title = None
    axis.set_xlabel('Amplitude right legs') # cols are idx_y, right legs
    axis.set_ylabel('Amplitude left legs') # rows are idx_x, left legs
    axis.plot((0, len(actions_range_x)-1), (0, len(actions_range_y)-1), 'b')
    img_data = np.zeros((len(actions_range_x), len(actions_range_y)))
    axim = axis.imshow(img_data, origin='lower', cmap=pylab.cm.Greys)
    fig.colorbar(axim)
    
    # action line
    if with_actions:
        ox, sx = actions_range_x[0], len(actions_range_x)-1
        oy, sy = actions_range_y[0], len(actions_range_y)-1
        a_line = axis.plot([sx*(0.5-ox), sx*(0.5-ox)], [sy*(0.5-oy), sy*(1.0-oy)], 'r')[0]
        a_marker = axis.plot([sx*(0.5-ox)], [sy*(0.5-oy)], 'rs')[0]
    else:
        ox = sx = oy = sy = a_line = a_marker = None
    
    return fig, axis, axim, title, (a_line, a_marker, (ox,sx), (oy,sy))

def puppy_vid_inspected_trajectory(grp, step_width, loc_marker, epoch_idx, obs_offset):
    loc_x = grp['puppyGPS_x'][obs_offset+step_width*epoch_idx+step_width-1]
    loc_y = grp['puppyGPS_y'][obs_offset+step_width*epoch_idx+step_width-1]
    loc_marker.set_data([loc_x], [loc_y])
    return loc_marker
