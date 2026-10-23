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
MODULES = (
    'ess.isissans',
    'ess.loki',
    'ess.loki.workflow',
    'ess.sans',
    'ess.sans.workflow',
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


if __name__ == '__main__':
    test_has_version()
    test_import_works_without_optional_widget_packages()
