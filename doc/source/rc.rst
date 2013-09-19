
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
    :members:
    :show-inheritance:

.. autoclass:: PlainRLS
    :members:
    :special-members: __call__

.. autoclass:: StabilizedRLS
    :members:
    :show-inheritance:


.. autofunction:: dense_w_in
.. autofunction:: sparse_w_in
.. autofunction:: dense_w_bias
.. autofunction:: sparse_reservoir
.. autofunction:: orthogonal_reservoir
.. autofunction:: chain_of_neurons
.. autofunction:: ring_of_neurons

.. autofunction:: reservoir_memory
.. autofunction:: query_reservoir_memory
.. autofunction:: find_radius_for_mc

