ess - Data reduction for ESS instrumentation
============================================

`European Spallation Source`_ (ESS) toolkit for neutron scattering data reduction powered by `scipp`_ and `scippneutron`_.
Provides ESS facility and instrument bespoke tools.
The ``ess`` module is part of the software stack for data reduction:

.. raw:: html
    :file: images/software-stack.svg

- ``scipp``, ``scippneutron``, and ``ess`` are Python packages that can be installed using ``conda``.
  Each package is released independently.
- ``ess-notebooks`` is a ``git`` repository containing Jupyter notebooks with examples and actual reduction workflows.
- Higher levels of the stack are optional, and it is absolutely possible to use ``scippneutron`` without ``ess``, or ``ess`` without ``ess-notebooks``.

New features may frequently be introduced on the top of the software stack, for example in ``ess-notebooks``.
Depending on the feature this may then gradually move to lower levels.
This involves a "filtering" process, since scope and contribution guidelines are different the lower the level in the stack.

.. _European Spallation Source: https://europeanspallationsource.se
.. _scipp: https://scipp.github.io
.. _scippneutron: https://scipp.github.io/scippneutron/index.html

Documentation
=============

.. toctree::
   :maxdepth: 3
   :caption: Getting started

   getting-started/installation

.. toctree::
   :maxdepth: 3
   :caption: Instruments

   instruments/amor/amor
   instruments/loki/loki
   instruments/external/external

.. toctree::
   :maxdepth: 3
   :caption: Techniques

   techniques/diffraction/diffraction
   techniques/reflectometry/reflectometry
   techniques/sans/sans
   techniques/wfm/wfm

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utilities/utilities

.. toctree::
   :caption: Developer documentation
   :maxdepth: 3

   developer/style-guide
   developer/getting-started

.. toctree::
   :caption: About
   :maxdepth: 3

   about/release-notes
