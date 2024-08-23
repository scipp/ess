# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ess import isissans, loki, sans


def test_has_version():
    assert hasattr(isissans, '__version__')
    assert hasattr(loki, '__version__')
    assert hasattr(sans, '__version__')


if __name__ == '__main__':
    test_has_version()
