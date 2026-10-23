# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Tests of package integrity.

Note that additional imports need to be added for repositories that
contain multiple packages.
"""

import subprocess
import sys

from ess import bifrost, cspec, miracles, spectroscopy, trex


def test_has_version():
    assert hasattr(bifrost, '__version__')
    assert hasattr(cspec, '__version__')
    assert hasattr(miracles, '__version__')
    assert hasattr(trex, '__version__')
    assert hasattr(spectroscopy, '__version__')


# Not runtime dependencies, so no module may need them at import time.
OPTIONAL_WIDGET_PACKAGES = ('ipympl', 'ipywidgets', 'pythreejs')
PACKAGES = (
    'ess.bifrost',
    'ess.cspec',
    'ess.miracles',
    'ess.spectroscopy',
    'ess.trex',
)
# Notebook-only modules that import widget packages at module level.
# Their submodules are skipped as well.
WIDGET_MODULES = ()


def test_import_works_without_optional_widget_packages():
    # A fresh interpreter, since other tests in this process import these packages.
    # `None` in sys.modules makes an import fail as if the package was not installed.
    # walk_packages ignores ImportError when it imports a subpackage to list its
    # contents, but every name it yields is imported explicitly, so none is hidden.
    code = f"""
import importlib, pkgutil, sys
for name in {OPTIONAL_WIDGET_PACKAGES!r}:
    sys.modules[name] = None
widget_modules = tuple(module + '.' for module in {WIDGET_MODULES!r})
for root in {PACKAGES!r}:
    package = importlib.import_module(root)
    for module in pkgutil.walk_packages(package.__path__, root + '.'):
        if not (module.name + '.').startswith(widget_modules):
            importlib.import_module(module.name)
"""
    result = subprocess.run(  # noqa: S603 - fixed command, no external input
        [sys.executable, '-c', code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# This is for CI package tests. They need to run tests with minimal dependencies,
# that is, without installing pytest. This code does not affect pytest.
if __name__ == '__main__':
    test_has_version()
    test_import_works_without_optional_widget_packages()
