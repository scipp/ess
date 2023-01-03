# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import os
import pytest


@pytest.fixture(autouse=True)
def set_env_variables():
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
