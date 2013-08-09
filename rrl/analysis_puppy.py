"""
Analysis tools, specific for the *Puppy* experiment.
"""

import pylab

def puppy_plot_trajectory(analysis, axis, episode, **kwargs):
    """Plot the trajectory of an episode
    """
    gps_x = analysis[episode]['puppyGPS_x'][149::150]
    gps_y = analysis[episode]['puppyGPS_y'][149::150]
    col = kwargs.pop('color', 'k')
    axis.plot(gps_x, gps_y, '*', color=col, **kwargs)
    axis.plot(gps_x, gps_y, color=col, **kwargs)
    return axis

def puppy_plot_all_trajectories(analysis, axis):
    """Plot all trajectories in ``analysis`` into ``axis``.
    """
    gps_x = analysis.get_data('puppyGPS_x')
    gps_y = analysis.get_data('puppyGPS_y')
    
    N = len(gps_x)-1
    for x, y in zip(gps_x, gps_y):
        col = 0.75 - (0.75 * (idx - 1))/N
        axis.plot(x, y, color=col)
    
    return ax
