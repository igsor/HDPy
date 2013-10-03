"""
For puppy experiment analysis, snapshot functions are implemented in a
similar fashion as for the :ref:`ePuck robot <epuck>`. However, for
Puppy, the action is assumed to be two dimensional. The action snapshot
is hence an image (2d plot). Through :py:func:`puppy_plot_action`, the
figure is plotted at a specific state (identified by the epoch index
of a recorded episode). Furthermore, the overall trajectory and the
location of the inspected states can be plotted through 
:py:func:`puppy_plot_inspected_trajectory`.
This method can either be used at some isolated states (with the
mentioned methods) or in a video-like fashion. For the latter case,
:py:class:`PuppyActionVideo` implements the necessary routines.

The environment plotting can be managed through the functions
:py:func:`puppy_plot_linetarget`, :py:func:`puppy_plot_locationtarget`
and :py:func:`puppy_plot_landmarks`, dependent on the training target
(as defined in :ref:`plants_puppy`). For plotting the robot's trajectory
the functions :py:func:`puppy_plot_trajectory` and
:py:func:`puppy_plot_all_trajectories` can be used.

"""
import pylab
import numpy as np
import itertools
import warnings
from puppy import SENSOR_NAMES

def plot_trajectory(analysis, axis, episode, step_width=1, offset=0, legend=True, **kwargs):
    """Plot the trajectory of an episode
    """
    gps_x = analysis[episode]['puppyGPS_x'][offset+step_width-1::step_width]
    gps_y = analysis[episode]['puppyGPS_y'][offset+step_width-1::step_width]
    if step_width > 1:
        gps_x = np.concatenate(([analysis[episode]['puppyGPS_x'][offset]], gps_x))
        gps_y = np.concatenate(([analysis[episode]['puppyGPS_y'][offset]], gps_y))

    col = kwargs.pop('color', 'k')
    label = kwargs.pop('label', 'Trajectory')
    axis.plot(gps_x, gps_y, color=col, label=label, linewidth=3, **kwargs)
    axis.axis('equal')
    if legend:
        axis.plot(gps_x[0], gps_y[0], 'ks', label='Start')
        axis.plot(gps_x[-1], gps_y[-1], 'kv', label='End')
    
    return axis

def plot_all_trajectories(analysis, axis, step_width=1, **kwargs):
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

def plot_linetarget(axis, origin=(2.0, 0.0), direction=(1.0, 1.0), range_=(-5.0, 5.0)):
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

def plot_locationtarget(axis, target=(4.0, 4.0), distance=0.5, **kwargs):
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

def plot_landmarks(axis, landmarks, **kwargs):
    """Plot markers at ``landmark`` locations in ``axis``."""
    color = kwargs.pop('color', 'k')
    lbl = kwargs.pop('label', '')
    marker = kwargs.pop('marker','^')
    for x, y in landmarks:
        axis.plot([x], [y], marker=marker, color=color, label=lbl, **kwargs)
    return axis

def _action_eval(grp, reservoir, critic, trg_epoch, obs_offset, step_width, actions_range_x, actions_range_y):
    """Evaluate a set of two-dimensional actions [``action_range_x``,
    ``actions_range_y``] at a specific state ``trg_epoch`` and return
    the matrix of predicted returns.
    
    ``grp``
        Observed data of the underlying experiment. Usually a
        :py:class:`H5CombinedGroup` or [HDF5]_ group (e.g. through
        :py:class:`Analysis`).
    
    ``critic``
        :py:func:`critic` instance to be used for evaluation for a
        certain critic input (action and state).
    
    ``reservoir``
        Reservoir to be used. Note that this must be the same instance
        as used in ``critic``.
    
    ``obs_offset``
        Offset between robot observations (e.g. GPS) and reinforcement
        learning data (i.e. actions). For offline data, the offset is
        one epoch (i.e. ``step_width``), for online data, it is zero.
    
    ``step_width``
        Number of observations per epoch. In terms of :py:mod:`PuPy`,
        this is the control period over the sensor polling period.
    
    
    """
    reservoir.reset()
    reservoir.states = np.atleast_2d(grp['x_curr'][trg_epoch-1, :reservoir.get_output_dim()])
    
    # evaluate actions
    # Note: epoch is one step ahead (of a_curr, same time as a_next)!
    # Note: sensor values are shifted w.r.t a_curr by obs_offset
    s_curr = dict([(sensor, grp[sensor][obs_offset+step_width*(trg_epoch-1):obs_offset+trg_epoch*step_width]) for sensor in SENSOR_NAMES])
    a_ret = np.zeros((len(actions_range_x), len(actions_range_y)))
    
    actions_iter = itertools.product(range(len(actions_range_x)), range(len(actions_range_y)))
    for idx_x, idx_y in actions_iter:
        action_candidate = np.atleast_2d((actions_range_x[idx_x], actions_range_y[idx_y])).T
        j_curr = critic(s_curr, action_candidate, simulate=True)
        a_ret[idx_x, idx_y] = j_curr[0, 0]
        #print actions_range_x[idx_x], actions_range_y[idx_y], j_curr[0, 0]
    
    return a_ret

def plot_action(analysis, episode, critic, reservoir, inspect_epochs, actions_range_x, actions_range_y, step_width, obs_offset, epoch_actions=None):
    """Along a trajectory ``episode`` of a conducted experiment given
    by ``analysis``, plot the predicted return over a 2D-action at some
    fixed states. For each of the states (given by ``inspect_epochs``),
    a figure is created including the return prediction as an image
    (i.e. 2D).
    
    ``analysis``
        :py:class:`Analysis` instance containing the experimental data.
    
    ``episode``
        Episode which is analysed.
    
    ``critic``
        :py:func:`critic` instance to be used for evaluation for a
        certain critic input (action and state).
    
    ``reservoir``
        Reservoir to be used. Note that this must be the same instance
        as used in ``critic``.
    
    ``inspect_epochs``
        Epochs numbers for which the predicted actions should be
        plotted.
    
    ``actions_range_x``
        Action range in the first dimension. The return is predicted
        for any combination of ``actions_range_x`` and
        ``actions_range_y``.
    
    ``actions_range_y``
        Action range in the second dimension. The return is predicted
        for any combination of ``actions_range_x`` and
        ``actions_range_y``.
    
    ``step_width``
        Number of observations per epoch. In terms of :py:mod:`PuPy`,
        this is the control period over the sensor polling period.
    
    ``obs_offset``
        Offset between robot observations (e.g. GPS) and reinforcement
        learning data (i.e. actions). For offline data, the offset is
        one epoch (i.e. ``step_width``), for online data, it is zero.
    
    ``epoch_actions``
        A list of actually executed actions (as tuple), for each
        inspected epoch. The action is indicated in the plot by a
        marker. The argument or list items may be :py:const:`None`,
        in which case nothing is plotted.
    
    """
    grp = analysis[episode]
    if epoch_actions is None:
        epoch_actions = [None] * len(inspect_epochs)
    
    for trg_epoch, actions in zip(inspect_epochs, epoch_actions):
        
        # simulate the actions
        a_ret = _action_eval(grp, reservoir, critic, trg_epoch, obs_offset, step_width, actions_range_x, actions_range_y)
        
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

def plot_inspected_trajectory(analysis, episode_idx, step_width, axis, inspect_epochs, obs_offset):
    """Plot the robot trajectory of the experiment ``episode_idx``
    found in ``analysis`` and a marker at all ``inspect_epochs``. This
    function was created to support :py:func:`puppy_plot_action` by
    giving an overview over the whole path.
    
    ``axis``
        plotting canvas.
    
    ``step_width``
        Number of observations per epoch. In terms of :py:mod:`PuPy`,
        this is the control period over the sensor polling period.
    
    ``obs_offset``
        Offset between robot observations (e.g. GPS) and reinforcement
        learning data (i.e. actions). For offline data, the offset is
        one epoch (i.e. ``step_width``), for online data, it is zero.
    
    """
    puppy_plot_trajectory(analysis, axis, episode_idx, step_width, color='b', offset=obs_offset)
    trg_x = [analysis[episode_idx]['puppyGPS_x'][obs_offset + step_width*trg_epoch+step_width-1] for trg_epoch in inspect_epochs]
    trg_y = [analysis[episode_idx]['puppyGPS_y'][obs_offset + step_width*trg_epoch+step_width-1] for trg_epoch in inspect_epochs]
    axis.plot(trg_x, trg_y, 'k*', label='Inspected states')
    return axis

class ActionVideo:
    """Set up a structure such that the predicted return over 2D
    actions can be successively plotted in the same figure.
    
    .. todo::
        The selected action isn't displayed correctly (offset?)
    
    ``data``
        Observed data of the underlying experiment. Usually a
        :py:class:`H5CombinedGroup` or [HDF5]_ group (e.g. through
        :py:class:`Analysis`).
    
    ``critic``
        :py:func:`critic` instance to be used for evaluation for a
        certain critic input (action and state).
    
    ``reservoir``
        Reservoir to be used. Note that this must be the same instance
        as used in ``critic``.
    
    ``actions_range_x``
        Action range in the first dimension. The return is predicted
        for any combination of ``actions_range_x`` and
        ``actions_range_y``.
    
    ``actions_range_y``
        Action range in the second dimension. The return is predicted
        for any combination of ``actions_range_x`` and
        ``actions_range_y``.
    
    ``step_width``
        Number of observations per epoch. In terms of :py:mod:`PuPy`,
        this is the control period over the sensor polling period.
    
    ``obs_offset``
        Offset between robot observations (e.g. GPS) and reinforcement
        learning data (i.e. actions). For offline data, the offset is
        one epoch (i.e. ``step_width``), for online data, it is zero.
    
    ``with_actions``
        Plot markers and lines between them which represent the
        actually selected action.
    
    """
    def __init__(self, data, critic, reservoir, actions_range_x, actions_range_y, step_width, obs_offset, with_actions=True):
        
        # Basic figure
        self.fig = None
        self.title = None
        self.axis = None
        self.axis_image = None
        
        # Actions
        self.with_actions = with_actions
        self.actions_nrm = ((None, None), (None, None))
        self.actions_line = None
        self.actions_marker = None
        
        # Experiment data
        self.data = data
        self.critic = critic
        self.reservoir = reservoir
        self.step_width = step_width
        self.obs_offset = obs_offset
    
    def draw_init(self, fig=None):
        """Set up the initial video figure. A new figure is created
        unless one is provided in ``fig``.
        """
        if fig is None:
            fig = pylab.figure()
            
        # Create the figure
        self.fig = fig
        self.title = self.fig.suptitle('Expected Return per action')
        
        # Configure the axis
        self.axis = self.fig.add_subplot(111)
        self.axis.set_xticks(range(len(self.actions_range_y)))
        self.axis.set_xticklabels(self.actions_range_y)
        self.axis.set_yticks(range(len(self.actions_range_x)))
        self.axis.set_yticklabels(self.actions_range_x)
        self.axis.set_xlabel('Amplitude right legs') # cols are idx_y, right legs
        self.axis.set_ylabel('Amplitude left legs') # rows are idx_x, left legs
        
        # Plot the diagonal
        self.axis.plot((0, len(self.actions_range_x)-1), (0, len(self.actions_range_y)-1), 'b')
        
        # Prepare the image
        img_data = np.zeros((len(actions_range_x), len(actions_range_y)))
        self.axis_image = self.axis.imshow(img_data, origin='lower', cmap=pylab.cm.Greys)
        self.fig.colorbar(self.axis_image)
        
        # action line
        if self.with_actions:
            ox, sx = self.actions_range_x[0], len(self.actions_range_x)-1
            oy, sy = self.actions_range_y[0], len(self.actions_range_y)-1
            self.actions_nrm = ((ox, sx), (oy, sy))
            self.actions_line = self.axis.plot([sx*(0.5-ox), sx*(0.5-ox)], [sy*(0.5-oy), sy*(1.0-oy)], 'r')[0]
            self.actions_marker = self.axis.plot([sx*(0.5-ox)], [sy*(0.5-oy)], 'rs')[0]
        
        return self
    
    def draw_step(self, epoch, actions=None):
        """Update the figure by showing the action plot for ``epoch``.
        If `with_actions` is set, a list of actions to be plotted should
        be present in ``actions``.
        """
        # evaluate the actions
        a_ret = _action_eval(
            self.data,
            self.reservoir,
            self.critic,
            epoch,
            self.obs_offset,
            self.step_width,
            self.actions_range_x,
            self.actions_range_y
        )
        
        # update plot
        self.axis_image.set_data(a_ret)
        self.axis_image.set_clim(vmin=a_ret.min(), vmax=a_ret.max())
        self.axis_image.changed()
        
        # update action line and marker
        if self.with_actions:
            if actions is None:
                warnings.warn('with_actions set but no actions provided')
            else:
                ox, sx = self.actions_nrm[0]
                oy, sy = self.actiosn_nrm[1]
                actions[:, 0] = (actions[:, 0] - oy) * sy
                actions[:, 1] = (actions[:, 1] - ox) * sx
                self.actions_line.set_data((actions[:, 1], actions[:, 0]))
                self.actions_marker.set_data(([actions[-1, 1]], [actions[-1, 0]]))
        
        return self
    
    def draw_trajectory(self, loc_marker, epoch_idx):
        """Update the marker of the current state in a trajectory plot.
        The current state is read from *data* at ``epoch_idx``, the
        marker plot given in ``loc_marker``.
        """
        loc_x = self.data['puppyGPS_x'][self.obs_offset+self.step_width*epoch_idx+self.step_width-1]
        loc_y = self.data['puppyGPS_y'][self.obs_offset+self.step_width*epoch_idx+self.step_width-1]
        loc_marker.set_data([loc_x], [loc_y])
        return self


## DEPRECATED ##

def puppy_plot_trajectory(*args, **kwargs):
    """Alias of plot_trajectory.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_trajectory` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_trajectory' instead")
    return plot_trajectory(*args, **kwargs)

def puppy_plot_all_trajectories(*args, **kwargs):
    """Alias of plot_all_trajectories.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_all_trajectories` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_all_trajectories' instead")
    return plot_all_trajectories(*args, **kwargs)

def puppy_plot_linetarget(*args, **kwargs):
    """Alias of plot_linetarget.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_linetarget` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_linetarget' instead")
    return plot_linetarget(*args, **kwargs)

def puppy_plot_locationtarget(*args, **kwargs):
    """Alias of plot_locationtarget.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_locationtarget` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_locationtarget' instead")
    return plot_locationtarget(*args, **kwargs)

def puppy_plot_landmarks(*args, **kwargs):
    """Alias of plot_landmarks.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_landmarks` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_landmarks' instead")
    return plot_landmarks(*args, **kwargs)

def puppy_plot_action(*args, **kwargs):
    """Alias of plot_action.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_action` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_action' instead")
    return plot_action(*args, **kwargs)

def puppy_plot_inspected_trajectory(*args, **kwargs):
    """Alias of plot_inspected_trajectory.
    
    .. deprecated:: 1.0
        Use :py:func:`plot_inspected_trajectory` instead
        
    """
    warnings.warn("This function is deprecated. Use 'plot_inspected_trajectory' instead")
    return plot_inspected_trajectory(*args, **kwargs)

def puppy_vid_init(actions_range_x, actions_range_y, with_actions=True):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`PuppyActionVideo` instead
    
    """
    warnings.warn('deprecated, use PuppyActionVideo instead')
    fig = pylab.figure()
    axis = fig.add_subplot(111)
    axis.set_xticks(range(len(actions_range_y)))
    axis.set_xticklabels(actions_range_y)
    axis.set_yticks(range(len(actions_range_x)))
    axis.set_yticklabels(actions_range_x)
    title = fig.suptitle('Expected Return per action')
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
    
    return fig, axis, axim, title, (a_line, a_marker, (ox, sx), (oy, sy))

def puppy_vid_action(image, (a_line, a_marker, px, py), grp, critic, reservoir, epoch, actions_range_x, actions_range_y, step_width, obs_offset, actions=None):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`PuppyActionVideo` instead
        
    """
    warnings.warn('deprecated, use PuppyActionVideo instead')
    a_ret = _action_eval(grp, reservoir, critic, epoch, obs_offset, step_width, actions_range_x, actions_range_y)
    
    # update plot
    image.set_data(a_ret)
    image.set_clim(vmin=a_ret.min(), vmax=a_ret.max())
    image.changed()
    
    if actions is not None:
        actions[:, 0] = (actions[:, 0] - py[0]) * py[1]
        actions[:, 1] = (actions[:, 1] - px[0]) * px[1]
        a_line.set_data((actions[:, 1], actions[:, 0]))
        a_marker.set_data(([actions[-1, 1]], [actions[-1, 0]]))
    
    return image

def puppy_vid_inspected_trajectory(grp, step_width, loc_marker, epoch_idx, obs_offset):
    """
    
    .. deprecated:: 1.0
        Use :py:class:`PuppyActionVideo` instead
    
    """
    warnings.warn('deprecated, use PuppyActionVideo instead')
    loc_x = grp['puppyGPS_x'][obs_offset+step_width*epoch_idx+step_width-1]
    loc_y = grp['puppyGPS_y'][obs_offset+step_width*epoch_idx+step_width-1]
    loc_marker.set_data([loc_x], [loc_y])
    return loc_marker

