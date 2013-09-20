"""
The implementation of a custom policy is analogous to the creation
of a new :py:class:`Plant`. Here, the class :py:class:`Policy` is to be
subtyped, some of its methods are to be implemented. As with
:py:class:`Plant`, the normalization and action space dimensions are
automatically registered, in the later case through the default
constructor. Furthermore, the policy is reset at the beginning of a new
episode through :py:meth:`Policy.reset`.

The action itself is completely defined through the methods
:py:meth:`Policy.initial_action`, :py:meth:`Policy.update` and
:py:meth:`Policy.get_iterator`. The first gives a valid action, used
for initial behaviour (i.e. before the actor was in operation). The
other two define the behaviour during the experiment. After an action
has been selected, the :py:meth:`Policy.update` method is called,
which should note the new action and update internal structures. As with
the state, the action is passed as :math:`M \\times 1` vector. This
will be followed by a call to :py:meth:`Policy.get_iterator`, which
in turn produces the sequence of motor targets, as requested by
:py:class:`WebotsRobotMixin`.

"""
import policies_puppy as puppy
#import policies_epuck as epuck
