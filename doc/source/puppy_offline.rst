
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
and overwriting the :py:meth:`OfflineCollector._next_action_hook` for
action selection during the experiment. Hence, actions are chosen
randomly at all times, according to the respective schema. Note that
the action selection schema may have a huge influence on Critic training
later on.

.. literalinclude:: ../../test/puppy_offline_sampling_robot.py

With these two These two scripts, [Webots]_ can be executed and run for
some time. All sensor readouts and simulation metadata will be stored
in the file ``/tmp/puppy_offline_data.hdf5``. On this basis, a Critic
should be trained next.

$ webots_builder -c <robot.py> -s <supervisor.py> -t styrofoam -m fast /tmp/webots


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

When a Critic was trained, it is usually evaluated on a different
dataset. One possibility is to train the Critic on a part of the
training set and use the rest for testing. Then, it will be evaluated
on action sequences, sampled as in the training dataset. If this is not
desired, another set of offline data has to be acquired. In the
following code, such a set is created by a predefined action sequence.

Specifically, a main trajectory is defined, with some example
actions executed at every k'th steps. For this, the robot's state at the
k'th step must be identical for all example actions. As the robot cannot
easily be reset in [Webots]_, an easier approach is to revert the
simulation and keep the robot movement identical up to the k'th step.

Three scripts are given to achieve this task. The first script creates
a file which includes the action sequences and a reference to the one to
be executed next. The other two scripts are a robot and supervisor
controller for webots. Basically, an action sequence is loaded and
executed, the measurements stored in a file in the same fashion as in
the last section (:ref:`offline-data`). Hence, the file structure and
called functions are the same as before.


First, a number of action sequences is stored in a file at
``/tmp/example_sequence.hdf5``.

.. literalinclude:: ../../test/puppy_example_trajectory_sequence.py

To collect the simulation data, again a supervisor and robot controller
have to be created. As noted before, the simulation is to be reverted
(not restarted!) after an action sequence has finished. In this example,
this is implemented by two guards which react accordingly to a signal
from the robot.

.. literalinclude:: ../../test/puppy_example_trajectory_supervisor.py

Hence, the main logic is implemented in the robot controller. A special
case of an :py:class:`OfflineCollector` is defined, enforcing the action
to follow a specified sequence. If the sequence has ended, a signal is
sent to the supervisor. The action sequence is loaded from the HDF5
file, which was written before and the file updated such that all the
sequences will be executed. The initialization of the robot is then
analoguous to the previous section.

.. literalinclude:: ../../test/puppy_example_trajectory_robot.py

With the scripts set up, [Webots]_ can be executed. It automatically
quits after all trajectories have been handled. Note that the setup
of the policy, the number of initial steps and robot timings have been
set to the same values as in the training data collection process.

$ webots_builder -c <robot.py> -s <supervisor.py> -t styrofoam -m fast /tmp/webots

As with offline data acquisition, the robot data is written into a HDF5,
in this example at ``/tmp/example_data.hdf5``. Note that once this data
is available, it can be used for testing of several Critics (as for now,
all data is offline). Hence, the same process can be repeated for
several example trajectories to have a more representative testing
dataset.



.. _offline-analysis:

Critic analysis
^^^^^^^^^^^^^^^

If the example was followed until here, several files should be
available:

- ``/tmp/puppy_readout.pic``, the trained Critic's readout weights
- ``/tmp/puppy_reservoir.pic``, the Critic's reservoir
- ``/tmp/example_data.hdf5``, the testing dataset

With those, the Critic can finally be analyzed. To do so, the Critic
is executed on the testing dataset and then the result is plotted. The
first part works similar to the Critic's training. The testing data is
replayed, but this time the Critic is loaded instead of trained. The
following script achieves this, storing the evaluation result in
``/tmp/example_eval.hdf5``. As before, plant and policy are initialized,
then the reservoir and readout is loaded. Note that the readout training
is disabled. After creation of the :py:class:`PuppyHDP`, it is executed
on the testing data.

.. literalinclude:: ../../test/puppy_example_trajectory_eval.py

Now, the predicted return along the testing trajectory is stored in
``/tmp/example_eval.hdf5``. Based on this file, the Critic behaviour
can be analysed. It does not include the data collected during
simulation, hence the experiment is only completely described by also
considering ``/tmp/example_data.hdf5``. This is exactly what
:py:class:`H5CombineFile` is for.

Due to the initial behaviour of :py:class:`PuppyHDP` and
:py:class:`OfflinePuppy`, the datasets in the two files have a different
offset (indicated by ``obs_offset`` in the script). For the first epoch,
sensor data is available but no actions or reward. They are only stored
after the second step, hence are offset by one epoch (150 sensor samples
in this case). The predicted return is delayed even more, as it is not
stored during the whole initial phase (25 steps). The dataset can also
be thought of being aligned backwards.

The analysis script goes through all executions of the example
trajectory (one for each sample action) and orders them according to the
state in which the sample action execution started. For each of those
states, the sample actions are plotted as lines, colored with respect
to the respective predicted return. States itself are related by
plotting a circle, colored according to the median return of actions
executed from it.

.. literalinclude:: ../../test/puppy_offline_analysis.py

If it worked correctly, a plot should be generated which shows the
example trajectory, the sampled actions and states with the color
corresponding to the predicted return (darker is better).


.. image:: ../../data/puppy_offline_result.png
