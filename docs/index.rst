.. raw:: html

   <div style="display: block; margin-left: auto; margin-right: auto; width: 40%;">
      <img src="_static/logo.svg" width="100%" />
   </div>
   <style> .transparent {opacity:0; font-size:10px} </style>

.. role:: transparent

:transparent:`ess`
******************

.. raw:: html

   <div style="display: block;width: 100%;font-size:1.2em;font-style:italic;color:#5a5a5a;text-align: center;">
      Data reduction for ESS instrumentation
      </br></br>
   </div>


.. attention::

   The ``ess`` python package is being split-up into technique and instrument specific packages,
   and should be considered deprecated.
   The documentation here is kept for historical reasons.
   **Below, you will find links to the new packages.**


.. grid:: 2

    .. grid-item-card::  ESSdiffraction
        :link: https://scipp.github.io/essdiffraction/

        Diffraction data reduction

    .. grid-item-card::  ESSimaging
        :link: https://scipp.github.io/essimaging/

        Imaging data reduction

.. grid:: 2

    .. grid-item-card::  ESSpolarization
        :link: https://scipp.github.io/esspolarization/

        Polarization data reduction

    .. grid-item-card::  ESSreflectometry
        :link: https://scipp.github.io/essreflectometry/

        Reflectometry data reduction

.. grid:: 2

    .. grid-item-card::  ESSsans
        :link: https://scipp.github.io/esssans/

        SANS data reduction

    .. grid-item-card::  ESSspectroscopy
        :link: https://scipp.github.io/essspectroscopy/

        Spectroscopy data reduction

.. grid:: 2

    .. grid-item-card::  ESSnmx
        :link: https://scipp.github.io/essnmx/

        Data reduction for the NMX instrument

    .. grid-item-card::  ESSreduce
        :link: https://scipp.github.io/essreduce/

        Common tools for ESS data reduction

.. raw:: html

   <br><br><br><br>

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

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Getting started

   getting-started/installation

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Instruments

   instruments/amor/amor
   instruments/dream/dream
   instruments/loki/loki
   instruments/external/external

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Techniques

   techniques/diffraction/diffraction
   techniques/reflectometry/reflectometry
   techniques/sans/sans
   techniques/wfm/wfm

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Utilities

   utilities/utilities

.. toctree::
   :caption: Developer documentation
   :hidden:
   :maxdepth: 3

   developer/style-guide
   developer/getting-started

.. toctree::
   :caption: About
   :hidden:
   :maxdepth: 3

   about/release-notes
