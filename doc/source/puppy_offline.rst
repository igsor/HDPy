
Puppy offline workflow
======================

In summary, the workflow of offline puppy experiments is:

1. :ref:`Capture offline training data <offline-data>`
2. :ref:`Train a Critic on offline data <offline-critic-training>`
3. :ref:`Create example trajectories <offline-examples>`
4. :ref:`Simulate the Critic on an example trajectory <offline-analysis>`
5. :ref:`Evaluate the Critic on the example trajectory <offline-analysis>`

In this document, these steps will be discussed in detail.

Working offline has the advantage over working online that the
relatively slow part - the data acquisition - is executed once, then
several Critics can be trained on the same dataset. Since the simulation
is only invoked once for all Critics, this approach is much faster.
Also, since the dataset is the same for all Critics, a comparison of the
results is possible.

.. note::
    Some parameters are global over all scripts, for example the
    sampling period or file paths. It must be ensured that the exact
    same values are used throughout the whole process.



.. _offline-data:

Gathering offline data
^^^^^^^^^^^^^^^^^^^^^^

When working with [Webots]_, two scripts are required: a robot and a
supervisor. Note that this setup is fully described in the :py:mod:`PuPy`
documentation. For offline data acquisition, a supervisor is created
which resets the simulation whenever Puppy tumbles or leaves a
predefined arena.

.. literalinclude:: ../../test/puppy_offline_sampling_supervisor.py

The robot script is a bit more complex. The controller has to select
actions according to a predefined schema and store all data in a HDF5
file for later processing. To have the file in the correct format, the
class :py:class:`OfflineCollector` has to be used. It records all data
such that the simulation behaviour can be reproduced.

For the action selection mechanism, first a :py:class:`Policy` is
created. It defines the action and links it
to a motor target sequence, as explained in :ref:`plants-and-policies`.
In this case, the action is based on a gait and controlls the
amplitudes of the left and right legs. The procedure to create an
initial action is overwritten such that the initial action is randomly
chosen. The same is achieved by subtyping :py:class:`OfflineCollector`
and overwriting the :py:method:`OfflineCollector._next_action_hook` for
action selection during the experiment. Hence, actions are chosen
randomly at all times, according to the respective schema. Note that
the action selection schema may have a huge influence on Critic training
later on.

.. literalinclude:: ../../test/puppy_offline_sampling_robot.py

With these two These two scripts, [Webots]_ can be executed and run for
some time. All sensor readouts and simulation metadata will be stored
in the file ``/tmp/puppy_offline_data.hdf5``. On this basis, a Critic
should be trained next.



.. _offline-critic-training:

Critic training
^^^^^^^^^^^^^^^

For training, the :py:class:`Plant` must be specified and in case of
Puppy its ADHDP implementation in :py:class:`PuppyHDP`. Note that
although the :py:class:`Policy` is not in effect (as the selected
actions are fixed due to the offline setup), a valid instance must
be provided to the Critic. Here, the same one as for offline training is
initialized.

For Critic training, now also a reservoir and readout must be available,
as initialized in the example. Furthermore, the
:py:class:`PuPy.Normalization` is provided to the Critic, as during
offline data gathering the sensor data is not processed at all.

After the required objects have been created, they are bound together
in :py:class:`PuppyHDP`. It is also directed to store critic output
in the file ``/tmp/puppy_critic.hdf5``. Note that in this configuration,
sensor data is not copied, i.e. they are not included in the Critic's
data file, which is very convenient to save disk space.

Finally, the function :py:func:`puppy_offline_playback` is invoked. This
function replays the offline data such that the Critic sees it as if it
was run online in [Webots]_. Hence, the Critic is trained as in the
simulator. Only the data file has to be specified and optionally the
training set can be limited (in this case to 1000 episodes).

.. literalinclude:: ../../test/puppy_offline_replay.py

After the script was successfully executed, the trained critic is
available in three files:

- ``/tmp/puppy_critic.hdf5``
- ``/tmp/puppy_readout.pic``
- ``/tmp/puppy_reservoir.pic``

All data that is saved by the Critic is in the first file. The latter
two contain the reservoir and readout, as they cannot be stored in the
datafile. For further processing, the readout and reservoir files will
be required. The datafile mainly serves static training analysis.



.. _offline-examples:

Creating example trajectories and Critic evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




.. _offline-analysis:

Critic analysis
^^^^^^^^^^^^^^^


