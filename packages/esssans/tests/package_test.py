# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest

from ess import isissans, loki, sans


@pytest.mark.parametrize('pkg', [sans, loki, isissans])
def test_has_version(pkg):
    assert hasattr(pkg, '__version__')
