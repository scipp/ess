# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Generic

import scipp as sc

# TODO common name for "cell or supermirror", "Device"?
from .he3 import Cell


@dataclass
class SupermirrorTransmissionFunction(Generic[Cell]):
    """Wavelength-dependent transmission of a supermirror"""

    def __call__(self, wavelength: sc.Variable) -> sc.DataArray:
        """Return the transmission fraction for a given wavelength"""
        raise NotImplementedError
