
.. _puppy:

Puppy
=====

.. contents::

.. module:: HDPy

Introduction
------------

.. automodule:: HDPy.puppy

Example
-------

.. toctree::
   :maxdepth: 1
   
   puppy_offline
   puppy_online

Reference
---------

.. autoclass:: HDPy.puppy.PuppyHDP
    :members: new_episode, init_episode, _step, event_handler
    :show-inheritance:

.. autoclass:: HDPy.puppy.OfflineCollector
    :members: new_episode, __call__, _next_action_hook, event_handler
    :show-inheritance:

.. autofunction:: HDPy.puppy.offline_playback


.. autofunction:: HDPy.puppy.plot_trajectory

.. autofunction:: HDPy.puppy.plot_all_trajectories

.. autofunction:: HDPy.puppy.plot_linetarget

.. autofunction:: HDPy.puppy.plot_locationtarget

.. autofunction:: HDPy.puppy.plot_landmarks

.. autofunction:: HDPy.puppy.plot_action

.. autofunction:: HDPy.puppy.plot_inspected_trajectory

.. autoclass:: HDPy.puppy.ActionVideo
    :members:
