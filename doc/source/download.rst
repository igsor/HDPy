
Download
========

.. contents::

Installation
------------

Using `Pip Installs Python (Pip) <http://www.pip-installer.org/en/latest/index.html>`_,
simply type::

    pip install http://www.igsor.net/research/rrl/_downloads/latest.tar.gz

if you want to use the package from the webpage. If you have downloaded it yourself, use::

    pip install path/to/rrl.tar.gz

If you're using `distutils <http://docs.python.org/distutils/>`_, type::
    
    tar -xzf path/to/rrl.tgz        # extract files.
    cd rrl*                         # change into RRL directory.
    sudo python setup.py install    # install using distutils (as root).
    #rm -R .                        # remove source. If desired, uncomment this line.
    #cd .. && rmdir rrl*            # remove working directory. If desired, uncomment this line.

Make sure, `scipy <http://www.scipy.org/>`_ and `numpy <http://numpy.scipy.org/>`_ are
installed on your system.

.. todo::
    Also dependent on Oger and mdp

Getting started
---------------

Again, make sure that besides the `RRL <http://www.igsor.net/research/rrl/>`_, the packages `scipy <http://www.scipy.org/>`_
and `numpy <http://numpy.scipy.org/>`_ are installed on your system.

.. todo::
    Explanations and examples

Available downloads
-------------------

- :download:`RRL-0.1 dev <_downloads/rrl-0.1dev.tar.gz>` (latest)

.. sth:
    - :download:`This documentation (html) <dist/aiLib-0.1doc-html.tar.gz>` (current)
    - :download:`This documentation (pdf) <dist/aiLib-0.1doc.pdf>` (current)

License
-------

This project is released under the terms of the 3-clause BSD License. See the section
:ref:`license` for details.
