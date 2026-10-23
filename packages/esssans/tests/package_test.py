# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import subprocess
import sys

import pytest
from ess import isissans, loki, sans

"""Tests of package integrity.

Note that additional imports need to be added for repositories that
contain multiple packages.
"""


def test_has_version():
    assert hasattr(isissans, '__version__')
    assert hasattr(loki, '__version__')
    assert hasattr(sans, '__version__')


@pytest.mark.parametrize('module', ['ess.loki', 'ess.loki.workflow'])
def test_import_works_without_ipywidgets(module: str) -> None:
    # A fresh interpreter, since other tests in this process import ipywidgets.
    # `None` in sys.modules makes `import ipywidgets` fail as if not installed.
    code = f"import sys; sys.modules['ipywidgets'] = None; import {module}"
    result = subprocess.run(  # noqa: S603 - fixed command, no external input
        [sys.executable, '-c', code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


if __name__ == '__main__':
    test_has_version()
