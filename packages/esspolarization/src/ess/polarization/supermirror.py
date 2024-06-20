# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Generic

import sciline
import scipp as sc

from .types import PlusMinus, PolarizingElement, TransmissionFunction


class SupermirrorEfficiencyFunction(Generic[PolarizingElement]):
    def __call__(self, *, wavelength: sc.Variable) -> sc.DataArray:
        """Return the efficiency of a supermirror for a given wavelength"""
        raise NotImplementedError


@dataclass
class SupermirrorTransmissionFunction(TransmissionFunction[PolarizingElement]):
    """Wavelength-dependent transmission of a supermirror"""

    efficiency_function: SupermirrorEfficiencyFunction

    def __call__(
        self, *, wavelength: sc.Variable, plus_minus: PlusMinus
    ) -> sc.DataArray:
        """Return the transmission fraction for a given wavelength"""
        efficiency = self.efficiency_function(wavelength=wavelength)
        if plus_minus == 'plus':
            return 0.5 * (1 + efficiency)
        else:
            return 0.5 * (1 - efficiency)

    def apply(self, data: sc.DataArray, plus_minus: PlusMinus) -> sc.DataArray:
        """Apply the transmission function to a data array"""
        return self(wavelength=data.coords['wavelength'], plus_minus=plus_minus)


def get_supermirror_efficiency_function() -> (
    SupermirrorEfficiencyFunction[PolarizingElement]
):
    # TODO This will need some input parameters
    return SupermirrorEfficiencyFunction[PolarizingElement]()


def get_supermirror_transmission_function(
    efficiency_function: SupermirrorEfficiencyFunction[PolarizingElement],
) -> TransmissionFunction[PolarizingElement]:
    return SupermirrorTransmissionFunction[PolarizingElement](
        efficiency_function=efficiency_function
    )


supermirror_providers = (
    get_supermirror_efficiency_function,
    get_supermirror_transmission_function,
)


def SupermirrorWorkflow() -> sciline.Pipeline:
    return sciline.Pipeline(supermirror_providers)
