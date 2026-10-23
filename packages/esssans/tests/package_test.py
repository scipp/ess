# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import subprocess
import sys

from ess import isissans, loki, sans

"""Tests of package integrity.

Note that additional imports need to be added for repositories that
contain multiple packages.
"""


def test_has_version():
    assert hasattr(isissans, '__version__')
    assert hasattr(loki, '__version__')
    assert hasattr(sans, '__version__')


# Not runtime dependencies, so no module may need them at import time.
OPTIONAL_WIDGET_PACKAGES = ('ipympl', 'ipywidgets', 'pythreejs')
PACKAGES = (
    'ess.isissans',
    'ess.loki',
    'ess.sans',
)
# Notebook-only modules that import widget packages at module level.
# Their submodules are skipped as well.
WIDGET_MODULES = ('ess.loki.diagnostics',)


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


if __name__ == '__main__':
    test_has_version()
    test_import_works_without_optional_widget_packages()
