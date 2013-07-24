"""
Analysis tools, specific for the *epuck* experiment.
"""

def plot_all_trajectories(analysis, axis, key='loc'):
    """Plot trajectories of all episodes in ``analysis`` in the same
    plot ``axis``. The later an episode, the darker its trajectory is
    displayed. The trajectory data must be stored as ``key`` (default
    *loc*), a two-dimensional array. This function is intended to be
    used for analysis of **ePuck** experiments.
    """
    data = analysis.get_data(key)
    N = len(data)-1.0
    for idx, episode in enumerate(data):
        col = 0.75 - (0.75 * (idx - 1))/N
        axis.plot(episode[:, 0], episode[:, 1], color=str(col), label=str(idx))
    
    return axis

