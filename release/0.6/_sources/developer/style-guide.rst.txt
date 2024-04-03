Style guide
===========

We welcome contributions to this repository.
When contributing to this repository,
we encourage you to discuss the change you wish to make via issue, email,
or any other method with the owners of this repository before making a change.

Code contributions should be made via a github
`pull request <https://github.com/scipp/ess/pulls>`_ (PR).
Once a PR has been submitted, we conduct a review,
so please anticipate some discussion on anything you submit to ensure we can keep the code base of a good quality and maintainable.

Layout
------

This repository is ESS instrument centric,
and we therefore strongly prefer having all code organised into ``src/ess/{instrument}`` directories.

If your code is specific to the ESS facility,
but is intended to be ubiquitous across a class of instruments,
you may put it into a technique specific directory ``src/ess/{technique}``,
i.e ``src/ess/reflectometry``.

Code that is technique specific, but free from ESS facility considerations,
may be considered for inclusion in `scippneutron <https://github.com/scipp/scippneutron>`_.

Coding guidelines
-----------------

When writing a new module for the ``ess`` package,
please keep the following recommendations in mind,
as they will not only help standardize the code across the project,
but also speed up the reviewing process.

Prefer a functional design
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefer a functional design to a design that relies heavily on custom classes.
This means that you should, as much as possible,
keep your data inside native Scipp data structures
(``Variable``, ``DataArray`` and ``Dataset``).
For your data processing, you should then write functions that accept these objects,
and perform operations on them.

This will make it easier in the future to share your functions across different
instruments or techniques, and ensure that you can still use Scipp's in-built
functionalities such as plotting.

Logging
~~~~~~~

If you have functions in your workflow that represent important steps
(or costly operations) in your data processing, it is recommended to register the calls
to those functions via Scipp's logging framework.

More details can be found in the
`logging documentation <https://scipp.github.io/reference/logging.html>`_,
as well as in the
`Scipp documentation on logging <https://scipp.github.io/reference/logging.html>`_.

Testing
~~~~~~~

To ensure that the package is robust, authors must provide unit tests alongside code.
It is possible that future updates for Scipp or Scippneutron dependencies can break the code you contribute.
If we are aware of failing tests, we can provide future fixes and migrations for you.
Please avoid large data files, or any code requiring network access.
Test suites should be fast to execute.

Formatting
~~~~~~~~~~

Your Python code will be checked for errors and formatting when opening a PR.
We use the `flake8 <https://flake8.pycqa.org/en/latest/>`_ linter to check code quality,
and `yapf <https://github.com/google/yapf>`_ to enforce code formatting.
Make sure that running your code though these tools does not generate any output before
you push your changes.

From the top level directory, you can use

.. code-block:: sh

   >$ flake8 .
   >$ yapf --diff --recursive .

Jupyter notebooks style
-----------------------

The recommended way to add a data reduction workflow to ``ess`` is to create a Jupyter
notebook that outlines all the steps in the workflow.
These notebooks should be added in the ``docs/{technique}`` or ``docs/{instrument}``
folder, as they will then be built (and thus checked for errors) for every PR.

It is possible to have a notebook outlining the different steps in a workflow,
and having the entire workflow available inside a python wrapper function for convenience.
In this case, the workflow should be broken up into small parts (helper functions),
and those should then be called from the notebook, instead of duplicating the code that
carries out the operations between the notebook and the python codebase.

Using data files
~~~~~~~~~~~~~~~~

If your notebooks require data files, you should use the ``pooch`` utility to handle
file paths.
See `amor.data <https://github.com/scipp/ess/blob/main/src/ess/amor/data.py>`_ for an
example showing how it is used in practice.

References
~~~~~~~~~~

If you need to refer to scientific articles in your notebooks,
you should include a ``References`` section at the bottom of your notebook.

Inside this section, you should list the references in alphabatical order,
and have one markdown cell per entry.

To link from the citation in the notebook's text body to the reference,
place at the top of each reference cell a ``<div>`` tag with a unique ``id``,
and make sure to leave a empty line between the ``<div>`` and the reference.

References should be included with the following:

- List of author's last names and initials, with commas only between authors.
- The year of publication, in bold.
- The title of the article, in italics.
- The journal title, issue number and pages as a link, that redirects to the article DOI.

An example:

.. code-block:: md

   <div id='manasi2021'></div>

   Manasi I., Andalibi M. R., Atri R. S., Hooton J., King S. M., Edler K. J., **2021**,
   *Self-assembly of ionic and non-ionic surfactants in type IV cerium nitrate and urea based deep eutectic solvent*,
   [J. Chem. Phys. 155, 084902](https://doi.org/10.1063/5.0059238)

To cite the article in the text body,
use the Harvard (author-year) style in combination with a link, e.g.

.. code-block:: md

   [...] as was shown by [Manasi et al. (2021)](#manasi2021).


Documentation
-------------

Apart from workflows in Jupyter notebooks, please provide and update documentation.
This involves

- including python docstrings on your user facing functions
- providing code comments
- adding type-hints to your function arguments and return types (see `typing <https://docs.python.org/3/library/typing.html>`_)
- adding your functions to the API reference for your technique or instrument
- including any additional document (Jupyter notebook or .rst file) that helps explain or describe how your functions or module work

We will build and publish sphinx documentation located
`here <https://github.com/scipp/ess/tree/main/docs>`_.
