.. _installation:

Installation
============

The easiest way to install ``ess`` is using `conda <https://docs.conda.io>`_.
Packages from `Anaconda Cloud <https://conda.anaconda.org/scipp>`_ are available for Linux, macOS, and Windows.
It is recommended to create an environment rather than installing individual packages.

With the provided environment file
----------------------------------

1. Download :download:`ess.yml <../environments/ess.yml>`.
2. In a terminal run:

   .. code-block:: sh

      conda activate
      conda env create -f ess.yml
      conda activate ess
      jupyter lab

The ``conda activate`` ensures that you are in your ``base`` environment.
This will take a few minutes.
Above, replace ``ess.yml`` with the path to the download location you used to download the environment.
Open the link printed by Jupyter in a browser if it does not open automatically.

If you have previously installed ``ess`` with conda we nevertheless recommend creating a fresh environment rather than trying to ``conda update``.
You may want to remove your old environment first, e.g.,

.. code-block:: sh

   conda activate
   conda env remove -n ess

and then proceed as per instructions above.
The ``conda activate`` ensures that you are in your ``base`` environment.

Without the provided environment file
-------------------------------------

To create a new conda environment with ``ess``:

.. code-block:: sh

   conda create -n env_with_ess -c conda-forge -c scipp ess

To add ``ess`` to an existing conda environment:

.. code-block:: sh

   conda install -c conda-forge -c scipp ess

.. note::
   Installing ``ess`` on Windows requires ``Microsoft Visual Studio 2019 C++ Runtime`` installed.
   Visit `this page <https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0>`_ for the up to date version of the library.

After installation the modules ``ess``, ``scippneutron``, and ``scipp`` can be imported in Python.
Note that only the bare essential dependencies are installed.

To update or remove ``ess`` use `conda update <https://docs.conda.io/projects/conda/en/latest/commands/update.html>`_ and `conda remove <https://docs.conda.io/projects/conda/en/latest/commands/remove.html>`_.
