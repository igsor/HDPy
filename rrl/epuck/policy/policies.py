"""

.. todo::
    documentation

"""
from ...rl import Policy
import warnings
import numpy as np

class Heading(Policy):
    """ePuck policy with the heading as action.
    
    Due to historical reasons, it is up to the implementation of the
    robot to interprete the action (i.e. if it's considered relative
    or absolute).
    
    Note that since Webots is not used for ePuck simulation, the action
    sequence is reduced to a single item and hence not returned as list.
    This behaviour works fine with the :py:class:`Robot` class.
    
    """
    def __init__(self):
        super(Heading, self).__init__(action_space_dim=1)
        self.action = self.initial_action()
    
    def initial_action(self):
        """Return the initial action (0.0)."""
        return np.atleast_2d([0.0]).T
    
    def update(self, action_upd):
        """Update the action."""
        self.action = action_upd
    
    def get_iterator(self, time_start_ms, time_end_ms, step_size_ms):
        """Return the heading."""
        return self.action
    
    def reset(self):
        """Reset the action to the initial one (0.0)."""
        self.action = self.initial_action()

class HeadingRandInit(Heading):
    """ePuck policy with the heading as action and random
    initialization.
    
    The only difference to :py:class:`Heading` is that the initial
    action is not 0.0 but randomly sampled in [0, 2*pi].
    
    """
    def initial_action(self):
        """Sample an action and return it as initial one."""
        rnd = np.random.uniform(0.0, 2*np.pi)
        return np.atleast_2d([rnd]).T

class Trivial(Heading):
    """ePuck policy with the heading as action.
    
    .. deprecated:: 1.0
        Use :py:class:`Heading` instead
    
    """
    def __init__(self):
        warnings.warn("This class is deprecated. Use 'Heading' instead")
        super(Trivial, self).__init__()

class RandInit(HeadingRandInit):
    """ePuck policy with the heading as action and random
    initialization.
    
    .. deprecated:: 1.0
        Use :py:class:`HeadingRandInit` instead
    
    """
    def __init__(self):
        warnings.warn("This class is deprecated. Use 'HeadingRandInit' instead")
        super(RandInit, self).__init__()
