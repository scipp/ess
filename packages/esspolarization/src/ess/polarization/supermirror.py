# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Literal

import scipp as sc

from .types import Analyzer, Polarizer, PolarizingElement, TransmissionFunction


class SupermirrorEfficiencyFunction:
    def __call__(self, *, wavelength: sc.Variable) -> sc.DataArray:
        """Return the efficiency of a supermirror for a given wavelength"""
        raise NotImplementedError


@dataclass
class SupermirrorTransmissionFunction(TransmissionFunction[PolarizingElement]):
    """Wavelength-dependent transmission of a supermirror"""

    efficiency_function: SupermirrorEfficiencyFunction

    def __call__(
        self,
        *,
        wavelength: sc.Variable,
        plus_minus: Literal['plus', 'minus'],
    ) -> sc.DataArray:
        """Return the transmission fraction for a given wavelength"""
        efficiency = self.efficiency_function(wavelength=wavelength)
        if plus_minus == 'plus':
            return 0.5 * (1 + efficiency)
        else:
            return 0.5 * (1 - efficiency)


def supermirror_analyzer(
    func: SupermirrorTransmissionFunction[Analyzer],
) -> TransmissionFunction[Analyzer]:
    return func


def supermirror_polarizer(
    func: SupermirrorTransmissionFunction[Polarizer],
) -> TransmissionFunction[Polarizer]:
    return func
