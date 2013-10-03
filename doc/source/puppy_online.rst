
.. _puppy_online:

Puppy online workflow
=====================

.. note::
    The data recorded this way cannot be used to train another robot
    offline on the same dataset. This is because vital metadata for
    replay is not stored by the normal HDP implementation.

In the online setup, all comutations are done within a running [Webots]_
instance. For [Webots]_, a supervisor and robot controller script is
required, as documented in :py:mod:`PuPy`. In this example, the
simulation is to be reverted whenever the robot falls or leaves a
predefined arena. In this case, reverting the simulation is preferred
over respawning the robot, since this guarantees that the robot is
started in the same state in every episode.

.. literalinclude:: ../../test/puppy_online_supervisor.py

The robot controller is structured similar to the ones in the offline
case. First, the preliminaries for :py:class:`ADHDP` are to be prepared.
Hence, in the initialization the policy, plant and echo-state network
is created. Furthermore, an :math:`\epsilon`-greedy acting schema is
set up, by subtyping :py:class:`PuppyHDP` and specifying
:py:meth`ActorCritic._next_action_hook`. Once, the actor-critic instance
is ready in *acd*, the simulation is set up and finally run. In contrast
to the offline case, the Actor-Critic instance is combined with Webots,
as documented in :py:mod:`PuPy`.

.. literalinclude:: ../../test/puppy_online_robot.py

These two controllers can be loaded into webots and the simulation
executed. All observations will be stored in the file
``/tmp/puppy_online.hdf5``, the reservoir and readout are saved at
``/tmp/puppy_reservoir.pic`` and ``/tmp/puppy_readout.pic``.

The simulation is reverted once in a while, hence the controller script
will be terminated and reloaded several times. For the
controller to work, it must load the reservoir and readout if they
already exist. Note that the approach shown below saves the readout
before exiting. This will fail for large reservoirs, as the teardown
timeframe is limited by webots. In such a case, the readout weights
may be written into a file (and regained from it upon startup) at every
iteration (this can efficiently be done by means of a seperate HDF5
file).
