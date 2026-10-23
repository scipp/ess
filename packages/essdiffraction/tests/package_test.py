# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Tests of package integrity.

Note that additional imports need to be added for repositories that
contain multiple packages.
"""

import subprocess
import sys

from ess import diffraction as pkg


def test_has_version():
    assert hasattr(pkg, '__version__')


# Not runtime dependencies, so no module may need them at import time.
OPTIONAL_WIDGET_PACKAGES = ('ipympl', 'ipywidgets', 'pythreejs')
MODULES = (
    'ess.beer',
    'ess.beer.workflow',
    'ess.diffraction',
    'ess.dream',
    'ess.dream.workflows',
    'ess.magic',
    'ess.magic.workflow',
    'ess.powder',
    'ess.powder.workflow',
    'ess.snspowder',
)


def test_import_works_without_optional_widget_packages():
    # A fresh interpreter, since other tests in this process import these packages.
    # `None` in sys.modules makes an import fail as if the package was not installed.
    code = '; '.join(
        [
            'import sys',
            *(f'sys.modules[{name!r}] = None' for name in OPTIONAL_WIDGET_PACKAGES),
            *(f'import {module}' for module in MODULES),
        ]
    )
    result = subprocess.run(  # noqa: S603 - fixed command, no external input
        [sys.executable, '-c', code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# This is for CI package tests. They need to run tests with minimal dependencies,
# that is, without installing pytest. This code does not affect pytest.
if __name__ == '__main__':
    test_has_version()
    test_import_works_without_optional_widget_packages()
