
.. _plants-and-policies:

Plants and Policies
===================

.. contents::


Introduction
------------

.. module:: rrl

As described in :ref:`Reinforcement Learning <reinforcement-learning>`,
the learning problem formulation is achieved by specifying a
:py:class:`Plant` and a :py:class:`Policy`. For *Puppy* and *ePuck*,
some examples have already been implemented.

.. plant

To create a custom plant, basically the :py:class:`Plant` class has to
be subtyped. As the plant models the environment, it has to compute
a reward and state from sensor measurements. Together, they encode the
abstract learning target, a problem designer has in mind. The
implementation of a plant is quite straight-forward. The two functions
:py:meth:`Plant.state_input` and :py:meth:`Plant.reward` are called
whenever the state or reward is requested. They are expected to return
a vector (:math:`N \times 1`) and scalar, respectively. The state space
dimension :math:`N` may be announced through the plant's constructor and
later queried by calling :py:meth:`Plant.state_space_dim`. If the
plant is dependent on the episode, the :py:meth:`reset` method can be
implemented as well to reset the instance's internal state. Note that
the sensor values are not preprocessed, specifically not normalized.
To do so, a normalization instance (:py:class:`PuPy.Normalization`) is 
automatically registered at :py:attr:`Plant.normalization`. Note that
normalization is mandatory for :py:meth:`Plant.input_state`.

.. policy

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
the state, the action is passed as :math:`M \times 1` vector. This
will be followed by a call to :py:meth:`Policy.get_iterator`, which
in turn produces the sequence of motor targets, as requested by
:py:class:`WebotsRobotMixin`.


Reference
---------

.. autoclass:: Plant
    :members:

.. autoclass:: Policy
    :members:


.. _plants_puppy:

Puppy Plants
^^^^^^^^^^^^

.. autoclass:: rrl.puppy.plant.SpeedReward

.. autoclass:: rrl.puppy.plant.LineFollower

.. autoclass:: rrl.puppy.plant.TargetLocation

.. autoclass:: rrl.puppy.plant.TargetLocationLandmarks

.. autoclass:: rrl.puppy.plant.DiffTargetLocationLandmarks



.. _policies_puppy:

Puppy Policies
^^^^^^^^^^^^^^

.. GaitPolicy

.. automodule:: rrl.puppy.policy.policies


Examples:

.. autoclass:: rrl.puppy.policy.FRA

.. autoclass:: rrl.puppy.policy.LRA

.. autoclass:: rrl.puppy.policy.LRP

.. autoclass:: rrl.puppy.policy.IIAPFO


.. _plants_epuck:

ePuck Plants
^^^^^^^^^^^^

.. autoclass:: rrl.epuck.plant.CollisionAvoidanceFrontal
    :show-inheritance:

.. autoclass:: rrl.epuck.plant.CollisionAvoidanceSideways
    :show-inheritance:

.. autoclass:: rrl.epuck.plant.CollisionAvoidanceFull
    :show-inheritance:

.. autoclass:: rrl.epuck.plant.Attractor
    :show-inheritance:



.. _policies_epuck:

ePuck Policies
^^^^^^^^^^^^^^

.. autoclass:: rrl.epuck.policy.Heading
    :show-inheritance:

.. autoclass:: rrl.epuck.policy.HeadingRandInit
    :show-inheritance:
