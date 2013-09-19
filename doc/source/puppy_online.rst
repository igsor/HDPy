

Puppy online workflow
=====================

- setup
- robot
- supervisor
- analysis

.. note::
    The data recorded this way cannot be used to train another robot
    offline on the same dataset. This is because vital metadata for
    replay is not stored by the normal HDP implementation.



.. literalinclude:: ../../test/puppy_online_supervisor.py

.. literalinclude:: ../../test/puppy_online_robot.py

