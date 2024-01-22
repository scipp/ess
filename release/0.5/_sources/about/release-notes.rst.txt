.. _release-notes:

Release Notes
=============

v0.5.0 (April 2022)
-------------------

Features
~~~~~~~~

* Added resolution function for Amor and support for Orso file format in the ``reflectometry`` module `#115 <https://github.com/scipp/ess/pull/115>`_ (*Reflectometry*).
* Added footprint correction, super-mirror calibration, and normalisation between sample and supermirror measurements on a per-pixel & per-Q-bin level in the Amor workflow `#97 <https://github.com/scipp/ess/pull/97>`_ (*Reflectometry*).

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Andrew McCluskey :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.4.0 (February 2022)
----------------------

Breaking changes
~~~~~~~~~~~~~~~~

* When doing ``import ess``, all the submodules (``amor``, ``reflectometry``, and ``wfm``) are no longer directly available as ``ess.amor``. Instead, we now rely on simply doing ``from ess import amor`` or ``import ess.sans as sans`` `#102 <https://github.com/scipp/ess/pull/102>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.3.0 (February 2022)
----------------------

Features
~~~~~~~~

* Implemented a logging framework for reflectometry and Amor `#93 <https://github.com/scipp/ess/pull/93>`_.
* Added the ``sans.to_I_of_Q`` workflow to reduce SANS2D data, as well as notebooks that describe the workflow and illustrate its usage `#60 <https://github.com/scipp/ess/pull/60>`_.

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Wojciech Potrzebowski :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.2.0 (January 2022)
---------------------

Breaking changes
~~~~~~~~~~~~~~~~

* A major rewrite of the reflectometry code was undertaken `#61 <https://github.com/scipp/ess/pull/61>`_:

  * The reflectometry and Amor Data classes were removed in favour of a formalism consisting of free-functions that accept Scipp DataArrays and Datasets.
  * Moved the chopper class from the `wfm` submodule to free functions in its own ``choppers`` module.
  * The unit conversion (computing wavelength and Q) now use ``transform_coords``.
  * The Amor reduction notebook from ``ess-notebooks`` has been imported into the ``ess`` repository.

Contributors
~~~~~~~~~~~~

Owen Arnold :sup:`b, c`\ ,
Simon Heybrock :sup:`a`\ ,
Andrew McCluskey :sup:`a`\ ,
Samuel Jones :sup:`b`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.1.0 (September 2021)
-----------------------

This is the first official release of ``scipp/ess``.
The API may change without notice in future releases.

Features
~~~~~~~~

* Support for Amor data reduction
* Support for WFM data reduction (V20 and ODIN)
* Limited support for V20 Bragg-edge imaging

Contributors
~~~~~~~~~~~~

Matthew Andrew :sup:`b, c`\ ,
Owen Arnold :sup:`b, c`\ ,
Simon Heybrock :sup:`a`\ ,
Andrew McCluskey :sup:`a`\ ,
and Neil Vaytet :sup:`a`\

Contributing Organizations
--------------------------
* :sup:`a`\  `European Spallation Source ERIC <https://europeanspallationsource.se/>`_, Sweden
* :sup:`b`\  `Science and Technology Facilities Council <https://www.ukri.org/councils/stfc/>`_, UK
* :sup:`c`\  `Tessella <https://www.tessella.com/>`_, UK
