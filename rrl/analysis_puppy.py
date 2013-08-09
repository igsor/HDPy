"""
Analysis tools, specific for the *Puppy* experiment.
"""

import pylab
import numpy as np

def puppy_plot_trajectory(analysis, axis, episode, step_width=1, **kwargs):
    """Plot the trajectory of an episode
    """
    gps_x = analysis[episode]['puppyGPS_x'][step_width-1::step_width]
    gps_y = analysis[episode]['puppyGPS_y'][step_width-1::step_width]
    if step_width > 1:
        gps_x = np.concatenate(([analysis[episode]['puppyGPS_x'][0]], gps_x))
        gps_y = np.concatenate(([analysis[episode]['puppyGPS_y'][0]], gps_y))

    col = kwargs.pop('color', 'k')
    label = kwargs.pop('label', 'Trajectory')
    axis.plot(gps_x, gps_y, color=col, **kwargs)
    axis.plot(gps_x[0], gps_y[0], 'ks', label='Start')
    axis.plot(gps_x[-1], gps_y[-1], 'k^', label='End')
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
    lbl = kwargs.pop('label', '')
    axis.plot([target[0]], [target[1]], 'kD', **kwargs)
    if distance > 0.0:
        trg_field = pylab.Circle(target, distance, fill=fill, facecolor=color, linewidth=linewidth, label=lbl, **kwargs)
        axis.add_artist(trg_field)

    return axis

