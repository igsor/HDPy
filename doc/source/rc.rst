
Reservoir Computing
===================

.. contents::

Introduction
------------

.. automodule:: rrl.rc

Example
-------

.. todo::
    testcase


Reference
---------

.. module:: rrl

.. autoclass:: ReservoirNode
    :members: copy, input_dim, output_dim, reset, save, _post_update_hook, __call__

.. autoclass:: PlainRLS
    :members: train, __call__, save, stop_training

.. autoclass:: StabilizedRLS
    :members:
    :show-inheritance:


.. autofunction:: sparse_reservoir
.. autofunction:: dense_w_in
.. autofunction:: sparse_w_in
.. autofunction:: dense_w_bias
.. autofunction:: orthogonal_reservoir
.. autofunction:: chain_of_neurons
.. autofunction:: ring_of_neurons

.. autofunction:: reservoir_memory
.. autofunction:: find_radius_for_mc

