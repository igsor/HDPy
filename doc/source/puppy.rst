
.. _puppy:

Puppy
=====

.. contents::

.. module:: rrl

Introduction
------------

.. automodule:: rrl.puppy

Example
-------

.. toctree::
   :maxdepth: 1
   
   puppy_offline
   puppy_online

Reference
---------

.. autoclass:: rrl.puppy.PuppyHDP
    :members: new_episode, init_episode, _step, event_handler
    :show-inheritance:

.. autoclass:: rrl.puppy.OfflineCollector
    :members: new_episode, __call__, _next_action_hook, event_handler
    :show-inheritance:

.. autofunction:: rrl.puppy.offline_playback


.. autofunction:: rrl.puppy.plot_trajectory

.. autofunction:: rrl.puppy.plot_all_trajectories

.. autofunction:: rrl.puppy.plot_linetarget

.. autofunction:: rrl.puppy.plot_locationtarget

.. autofunction:: rrl.puppy.plot_landmarks

.. autofunction:: rrl.puppy.plot_action

.. autofunction:: rrl.puppy.plot_inspected_trajectory

.. autoclass:: rrl.puppy.ActionVideo
    :members:
