.. _release-notes:

Release Notes
=============

v0.3.0 (February 2022)
----------------------

Features
~~~~~~~~

* Added the ``sans.to_I_of_Q`` workflow to reduce SANS2D data, as well as notebooks that describe the workflow and illustrate its usage `#60 <https://github.com/scipp/ess/pull/60>`_.

Breaking changes
~~~~~~~~~~~~~~~~

Bugfixes
~~~~~~~~

Deprecations
~~~~~~~~~~~~

Stability, Maintainability, and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributors
~~~~~~~~~~~~

Simon Heybrock :sup:`a`\ ,
Wojciech Potrzebowski :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas Wynen :sup:`a`

v0.2.0 (January 2022)
---------------------

Features
~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

* A major rewrite of the reflectometry code was undertaken `#61 <https://github.com/scipp/ess/pull/61>`_:

  * The reflectometry and Amor Data classes were removed in favour of a formalism consisting of free-functions that accept Scipp DataArrays and Datasets.
  * Moved the chopper class from the `wfm` submodule to free functions in its own ``choppers`` module.
  * The unit conversion (computing wavelength and Q) now use ``transform_coords``.
  * The Amor reduction notebook from ``ess-notebooks`` has been imported into the ``ess`` repository.

Bugfixes
~~~~~~~~

Deprecations
~~~~~~~~~~~~

Stability, Maintainability, and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
