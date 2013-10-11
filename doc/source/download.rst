
Download
========

.. contents::

Installation
------------

Using `Pip Installs Python (Pip) <http://www.pip-installer.org/en/latest/index.html>`_,
simply type::

    pip install http://www.igsor.net/research/HDPy/_downloads/latest.tar.gz

if you want to use the package from the webpage. If you have downloaded it yourself, use::

    pip install path/to/HDPy.tar.gz

If you're using `distutils <http://docs.python.org/distutils/>`_, type::
    
    tar -xzf path/to/HDPy.tgz        # extract files.
    cd HDPy*                         # change into HDPy directory.
    sudo python setup.py install    # install using distutils (as root).
    #rm -R .                        # remove source. If desired, uncomment this line.
    #cd .. && rmdir HDPy*            # remove working directory. If desired, uncomment this line.

The project is also available on git, with the package and all supplementary data::

    git clone https://github.com/igsor/PuPy

Make sure, [numpy]_ and [scipy]_ are
installed on your system. For plotting, [matplotlib]_ is required.

- :download:`HDPy-1.0 <_downloads/HDPy-1.0.tar.gz>` (latest)

- :download:`This documentation (pdf) <_downloads/HDPy-1.0-doc.pdf>`

License
-------

This project is released under the terms of the 3-clause BSD License. See the section
:ref:`license` for details.
