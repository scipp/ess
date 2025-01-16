# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for Amor."""

import numpy as np
import scipp as sc
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source as orso_data_source
from orsopy.fileio.orso import Column, Orso, OrsoDataset

from ..reflectometry.orso import (
    OrsoDataSource,
    OrsoInstrument,
    OrsoIofQDataset,
    OrsoReduction,
)
from ..reflectometry.types import ReflectivityOverQ


def build_orso_instrument(events: ReflectivityOverQ) -> OrsoInstrument:
    """Build ORSO instrument metadata from intermediate reduction results for Amor.

    This assumes specular reflection and sets the incident angle equal to the computed
    scattering angle.
    """
    return OrsoInstrument(
        orso_data_source.InstrumentSettings(
            wavelength=orso_base.ValueRange(*_limits_of_coord(events, "wavelength")),
            incident_angle=orso_base.ValueRange(*_limits_of_coord(events, "theta")),
            polarization=None,  # TODO how can we determine this from the inputs?
        )
    )


def build_orso_iofq_dataset(
    iofq: ReflectivityOverQ,
    data_source: OrsoDataSource,
    reduction: OrsoReduction,
) -> OrsoIofQDataset:
    """Build an ORSO dataset for reduced I-of-Q data and associated metadata."""
    header = Orso(
        data_source=data_source,
        reduction=reduction,
        columns=[
            Column("Qz", "1/angstrom", "wavevector transfer"),
            Column("R", None, "reflectivity"),
            Column("sR", None, "standard deviation of reflectivity"),
            Column(
                "sQz",
                "1/angstrom",
                "standard deviation of wavevector transfer resolution",
            ),
        ],
    )
    iofq = iofq.hist()

    qz = iofq.coords["Q"].to(unit="1/angstrom", copy=False)
    if iofq.coords.is_edges("Q"):
        qz = sc.midpoints(qz)
    r = sc.values(iofq.data)
    sr = sc.stddevs(iofq.data)
    sqz = iofq.coords["Q_resolution"].to(unit="1/angstrom", copy=False)

    data = np.column_stack(tuple(map(_extract_values_array, (qz, r, sr, sqz))))
    data = data[np.isfinite(data).all(axis=-1)]
    ds = OrsoIofQDataset(OrsoDataset(header, data))
    ds.info.reduction.corrections = [
        "chopper ToF correction",
        "footprint correction",
        "supermirror calibration",
    ]
    return ds


def _extract_values_array(var: sc.Variable) -> np.ndarray:
    if var.variances is not None:
        raise sc.VariancesError(
            "ORT columns must not have variances. "
            "Store the uncertainties as standard deviations in a separate column."
        )
    if var.ndim != 1:
        raise sc.DimensionError(f"ORT columns must be one-dimensional, got {var.sizes}")
    return var.values


def _limits_of_coord(data: sc.DataArray, name: str) -> tuple[float, float, str] | None:
    if (coord := _get_coord(data, name)) is None:
        return None
    min_ = coord.min().value
    max_ = coord.max().value
    # Explicit conversions to float because orsopy does not like np.float* types.
    return float(min_), float(max_), _ascii_unit(coord.unit)


def _get_coord(data: sc.DataArray, name: str) -> sc.Variable | None:
    if name in data.coords:
        return sc.DataArray(data=data.coords[name], masks=data.masks)
    if (data.bins is not None) and (name in data.bins.coords):
        # Note that .bins.concat() applies the top-level masks
        events = data.bins.concat().value
        return sc.DataArray(data=events.coords[name], masks=events.masks)
    return None


def _ascii_unit(unit: sc.Unit) -> str:
    unit = str(unit)
    if unit == "Å":
        return "angstrom"
    return unit


providers = (build_orso_instrument, build_orso_iofq_dataset)
