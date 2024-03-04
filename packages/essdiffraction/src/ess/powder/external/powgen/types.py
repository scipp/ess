# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used by POWGEN.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from typing import NewType

import scipp as sc

# This is Mantid-specific and can probably be removed when the POWGEN
# workflow is removed.
DetectorInfo = NewType('DetectorInfo', sc.Dataset)
"""Mapping between detector numbers and spectra."""

del sc, NewType
