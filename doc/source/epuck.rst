
.. _epuck:

ePuck
=====

.. contents::

Introduction
------------

.. automodule:: rrl.epuck

Example
-------

.. literalinclude:: ../../test/epuck_online.py

.. image:: ../../data/doc/epuck_result.png

Reference
---------

.. module:: rrl

.. autoclass:: rrl.epuck.Robot
    :members: read_sensors, take_action, reset, reset_random, plot_trajectory

.. autoclass:: rrl.epuck.AbsoluteRobot
    :show-inheritance:

.. autofunction:: rrl.epuck.simulation_loop

.. autofunction:: rrl.epuck.epuck_plot_snapshot

.. autofunction:: rrl.epuck.epuck_plot_value_over_action

.. autofunction:: rrl.epuck.epuck_plot_all_trajectories
