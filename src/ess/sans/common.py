# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
from scipp.constants import g


def gravity_vector() -> sc.Variable:
    """
    Return a vector of 3 components, defining the magnitude and direction of the Earth's
    gravitational field.
    """
    return sc.vector(value=[0, -1, 0]) * g
