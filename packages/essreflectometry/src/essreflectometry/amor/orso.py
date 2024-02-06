# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for Amor."""
import scipp as sc
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source

from ..orso import OrsoInstrument
from ..types import Run, ThetaData, WavelengthData


def build_orso_instrument(
    events_in_wavelength: WavelengthData[Run], events_in_theta: ThetaData[Run]
) -> OrsoInstrument[Run]:
    """Build ORSO instrument metadata from intermediate reduction results for Amor.

    This assumes specular reflection and sets the incident angle equal to the computed
    scattering angle.
    """
    wavelength = events_in_wavelength.coords['wavelength']
    incident_angle = events_in_theta.coords['theta']
    # Explicit conversions to float because orsopy does not like np.float* types.
    return OrsoInstrument(
        data_source.InstrumentSettings(
            wavelength=orso_base.ValueRange(
                min=float(wavelength.min().value),
                max=float(wavelength.max().value),
                unit=_ascii_unit(wavelength.unit),
            ),
            incident_angle=orso_base.ValueRange(
                min=float(incident_angle.min().value),
                max=float(incident_angle.max().value),
                unit=incident_angle.unit,
            ),
            polarization=None,  # TODO how can we determine this from the inputs?
        )
    )


def _ascii_unit(unit: sc.Unit) -> str:
    unit = str(unit)
    if unit == 'Å':
        return 'angstrom'
    return unit


providers = (build_orso_instrument,)
