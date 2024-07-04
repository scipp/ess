# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# These fixtures cannot be found by pytest,
# if they are not defined in `conftest.py` under `tests` directory.
from contextlib import AbstractContextManager
from functools import partial

import pytest


@pytest.fixture()
def mcstas_2_deprecation_warning_context() -> partial[AbstractContextManager]:
    return partial(pytest.warns, DeprecationWarning, match="McStas")
