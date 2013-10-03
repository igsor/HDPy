
.. _reinforcement-learning:

Reinforcement Learning
======================

.. contents::

Introduction
------------

.. automodule:: rrl.rl

.. automodule:: rrl.hdp

Reference
---------

.. module:: rrl

.. autoclass:: Plant
    :members:
    :noindex:

.. autoclass:: Policy
    :members:
    :noindex:

.. autoclass:: ActorCritic
    :members: new_episode, __call__, init_episode, _step, _pre_increment_hook, _next_action_hook, save, load, set_normalization, set_alpha, set_gamma, set_momentum

.. autoclass:: Momentum
    :members: __call__

.. autoclass:: ConstMomentum
    :show-inheritance:

.. autoclass:: RadialMomentum
    :show-inheritance:

.. autoclass:: ADHDP
    :show-inheritance:
    :members: _critic_eval, _critic_deriv, init_episode, _step

.. autoclass:: CollectingADHDP
    :show-inheritance:
    :members: _pre_increment_hook

.. autoclass:: ActionGradient
    :show-inheritance:

.. autoclass:: ActionRecomputation
    :show-inheritance:

.. autoclass:: ActionBruteForce
    :show-inheritance:
