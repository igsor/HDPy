.. Reinforcement Reservoir Learning documentation master file, created by
   sphinx-quickstart on Wed May 22 19:50:46 2013.

Heuristic Dynamic Programming in Python
=======================================

.. automodule:: HDPy


This documentation gives an overview over the module's functionality,
gives an usage example and lists the interfaces. This order is kept
constant over all (i.e. most) pages. The first four pages
(:ref:`idx_basics`) list the basic interfaces and describe the methods
which implement Reservoir Computing and Reinforcement Learning. These
structures are independent on the experimental platform. 

This package was originally implemented for two platforms, the Puppy
and ePuck robots. The corresponding (and hence platform dependent) code
is documented in the second section (:ref:`idx_platforms`).

The third section (:ref:`idx_resources`) provides further information
and download and installation resources.

Note that some of the examples write files. In this case, the paths are
usually hardcoded and valid for a unix-like file tree. As data is
temporary, it is hence stored in ``/tmp``. When working on other
systems, the paths have to be adapted.

Furthermore, due to Python's magnificent online help, the interface
documentation is also available from within the interactive interpreter
(e.g. IPython):

>>> import HDPy
>>> help(HDPy)

.. note::
    The examples have been written for linux. As most of them include
    paths, they are also specified for a unix-like filesystem. On other
    systems, they have to be adapted. Also note that some of the paths
    may require adaptions, even on a linux machine (e.g. normalization
    data files).

Contents
--------

.. _idx_basics:

Basics
^^^^^^

.. toctree::
   :maxdepth: 1
   
   rc
   rl
   utils
   analysis

.. _idx_platforms:

Platforms
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   pp
   epuck
   puppy

.. _idx_resources:

Resources
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   todopg
   download
   license
   references


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

