# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for Amor."""
import numpy as np
import scipp as sc
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source as orso_data_source
from orsopy.fileio.orso import Column, Orso, OrsoDataset

from ..orso import OrsoDataSource, OrsoInstrument, OrsoIofQDataset, OrsoReduction
from ..types import NormalizedIofQ1D, QResolution, Run, ThetaData, WavelengthData


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
        orso_data_source.InstrumentSettings(
            wavelength=orso_base.ValueRange(
                min=float(wavelength.min().value),
                max=float(wavelength.max().value),
                unit=_ascii_unit(wavelength.unit),
            ),
            incident_angle=orso_base.ValueRange(
                min=float(incident_angle.min().value),
                max=float(incident_angle.max().value),
                unit=_ascii_unit(incident_angle.unit),
            ),
            polarization=None,  # TODO how can we determine this from the inputs?
        )
    )


def build_orso_iofq_dataset(
    iofq: NormalizedIofQ1D,
    sigma_q: QResolution,
    data_source: OrsoDataSource,
    reduction: OrsoReduction,
) -> OrsoIofQDataset:
    """Build an ORSO dataset for reduced I-of-Q data and associated metadata."""
    header = Orso(
        data_source=data_source,
        reduction=reduction,
        columns=[
            Column('Qz', '1/angstrom', 'wavevector transfer'),
            Column('R', None, 'reflectivity'),
            Column('sR', None, 'standard deviation of reflectivity'),
            Column(
                'sQz',
                '1/angstrom',
                'standard deviation of wavevector transfer resolution',
            ),
        ],
    )

    qz = iofq.coords['Q'].to(unit='1/angstrom', copy=False)
    if iofq.coords.is_edges('Q'):
        qz = sc.midpoints(qz)
    r = sc.values(iofq.data)
    sr = sc.stddevs(iofq.data)
    sqz = sigma_q.to(unit='1/angstrom', copy=False)
    data = (qz, r, sr, sqz)

    return OrsoIofQDataset(
        OrsoDataset(header, np.column_stack([_extract_values_array(d) for d in data]))
    )


def _extract_values_array(var: sc.Variable) -> np.ndarray:
    if var.variances is not None:
        raise sc.VariancesError(
            "ORT columns must not have variances. "
            "Store the uncertainties as standard deviations in a separate column."
        )
    if var.ndim != 1:
        raise sc.DimensionError(f"ORT columns must be one-dimensional, got {var.sizes}")
    return var.values


def _ascii_unit(unit: sc.Unit) -> str:
    unit = str(unit)
    if unit == 'Å':
        return 'angstrom'
    return unit


providers = (build_orso_instrument, build_orso_iofq_dataset)
