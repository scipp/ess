Getting Started
===============

Getting the source code
~~~~~~~~~~~~~~~~~~~~~~~

To get a copy of the code, you need to clone the git repository (either via SSH or HTTPS)
from `GitHub <https://github.com/scipp/ess>`_.
Once you have done this, go inside the top-level folder of the repository.

.. code-block:: bash

  cd ess

Creating a development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install all non-optional dependencies for a complete development environment,
we recommend creating a ``conda`` environment from a generated ``ess-developer.yml``
file.
We use the ``tools/metatoenv.py`` script to merge the dependencies in the
``conda/meta.yaml`` and ``developer-extra.yml`` files into a single
``ess-developer.yml`` file.

.. code-block:: bash

  python tools/metatoenv.py --dir=conda --env-file=ess-developer.yml \
    --channels=conda-forge,scipp,mantid --merge-with=developer-extra.yml
  conda env create -f ess-developer.yml
  conda activate ess-developer

Once you have activated the environment, you want to ``pip`` install it locally using

.. code-block:: bash

  python -m pip install -e .


Running the unit tests
~~~~~~~~~~~~~~~~~~~~~~

To run the unit tests, simply

.. code-block:: bash

  cd tests/
  python -m pytest

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

To build the documentation:

.. code-block:: bash

  cd docs
  sphinx-build -b html . build

This will build the documentation inside the ``docs/build`` folder.
If rebuilding the documentation is slow it can be quicker to remove the docs build
directory and start a fresh build.
