# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

import sciline
import scipp as sc

from .types import PlusMinus, PolarizingElement, TransmissionFunction


class SupermirrorEfficiencyFunction(Generic[PolarizingElement], ABC):
    """Base class for supermirror efficiency functions"""

    @abstractmethod
    def __call__(self, *, wavelength: sc.Variable) -> sc.DataArray:
        """Return the efficiency of a supermirror for a given wavelength"""


@dataclass
class SecondDegreePolynomialEfficiency(
    SupermirrorEfficiencyFunction[PolarizingElement]
):
    """
    Efficiency of a supermirror as a second-degree polynomial

    The efficiency is given by a * wavelength^2 + b * wavelength + c

    Parameters
    ----------
    a:
        Coefficient of the quadratic term, with unit of 1/angstrom^2
    b:
        Coefficient of the linear term, with unit of 1/angstrom
    c:
        Constant term, dimensionless
    """

    a: sc.Variable
    b: sc.Variable
    c: sc.Variable

    def __call__(self, *, wavelength: sc.Variable) -> sc.DataArray:
        """Return the efficiency of a supermirror for a given wavelength"""
        return (
            (self.a * wavelength**2).to(unit='', copy=False)
            + (self.b * wavelength).to(unit='', copy=False)
            + self.c.to(unit='', copy=False)
        )


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


def get_supermirror_transmission_function(
    efficiency_function: SupermirrorEfficiencyFunction[PolarizingElement],
) -> TransmissionFunction[PolarizingElement]:
    return SupermirrorTransmissionFunction[PolarizingElement](
        efficiency_function=efficiency_function
    )


def SupermirrorWorkflow() -> sciline.Pipeline:
    """
    Workflow for computing transmission functions for supermirror polarizing elements.
    """
    return sciline.Pipeline((get_supermirror_transmission_function,))
