# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for reflectometry.

The Sciline providers and types in this module largely ignore the metadata
of reference runs and only use the metadata of the sample run.
"""

import os
import platform
from datetime import datetime, timezone
from typing import NewType

import numpy as np
import scipp as sc
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source, orso, reduction
from orsopy.fileio.orso import Column, Orso, OrsoDataset

from .load import load_nx
from .types import (
    Beamline,
    Filename,
    Measurement,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRun,
)

OrsoCreator = NewType("OrsoCreator", orso_base.Person)
"""ORSO creator, that is, the person who processed the data."""

OrsoDataSource = NewType("OrsoDataSource", data_source.DataSource)
"""ORSO data source."""

OrsoExperiment = NewType("OrsoExperiment", data_source.Experiment)
"""ORSO experiment for the sample run."""

OrsoInstrument = NewType("OrsoInstrument", data_source.InstrumentSettings)
"""ORSO instrument settings for the sample run."""

OrsoIofQDataset = NewType("OrsoIofQDataset", orso.OrsoDataset)
"""ORSO dataset for reduced I-of-Q data."""

OrsoMeasurement = NewType("OrsoMeasurement", data_source.Measurement)
"""ORSO measurement."""

OrsoOwner = NewType("OrsoOwner", orso_base.Person)
"""ORSO owner of a measurement."""

OrsoReduction = NewType("OrsoReduction", reduction.Reduction)
"""ORSO data reduction metadata."""

OrsoSample = NewType("OrsoSample", data_source.Sample)
"""ORSO sample."""

OrsoSampleFilenames = NewType("OrsoSampleFilenames", list[orso_base.File])
"""Collection of filenames used to create the ORSO file"""

OrsoCorrectionList = NewType("OrsoCorrectionList", list[str])


def parse_orso_experiment(
    beamline: Beamline, measurement: Measurement
) -> OrsoExperiment:
    """Parse ORSO experiment metadata from raw NeXus data."""
    return OrsoExperiment(
        data_source.Experiment(
            instrument=beamline.name,
            facility=beamline.facility,
            title=measurement.title,
            start_date=measurement.start_time,
            proposalID=measurement.experiment_id,
            doi=measurement.experiment_doi,
            probe="neutron",
        )
    )


def parse_orso_owner(filename: Filename[SampleRun]) -> OrsoOwner:
    """Parse ORSO owner metadata from raw NeXus data."""
    (user,) = load_nx(filename, "NXentry/NXuser")
    return OrsoOwner(
        orso_base.Person(
            name=user["name"],
            contact=user["email"],
            affiliation=user.get("affiliation"),
        )
    )


def parse_orso_sample(filename: Filename[SampleRun]) -> OrsoSample:
    """Parse ORSO sample metadata from raw NeXus data."""
    (sample,) = load_nx(filename, "NXentry/NXsample")
    if not sample:
        return OrsoSample(data_source.Sample.empty())
    return OrsoSample(
        data_source.Sample(
            name=sample["name"],
            model=data_source.SampleModel(
                stack=sample.get("model", ""),
            ),
        )
    )


def orso_data_files(filename: Filename[SampleRun]) -> OrsoSampleFilenames:
    '''Collects names of files used in the experiment'''
    return [orso_base.File(file=os.path.basename(filename))]


def build_orso_measurement(
    sample_filenames: OrsoSampleFilenames,
    reference_filename: Filename[ReferenceRun],
    instrument: OrsoInstrument,
) -> OrsoMeasurement:
    """Assemble ORSO measurement metadata."""
    # TODO populate timestamp
    #      doesn't work with a local file because we need the timestamp of the original,
    #      SciCat can provide that
    if reference_filename:
        additional_files = [
            orso_base.File(
                file=os.path.basename(reference_filename), comment="supermirror"
            )
        ]
    else:
        additional_files = []
    return OrsoMeasurement(
        data_source.Measurement(
            instrument_settings=instrument,
            data_files=sample_filenames,
            additional_files=additional_files,
        )
    )


def build_orso_reduction(creator: OrsoCreator) -> OrsoReduction:
    """Construct ORSO reduction metadata.

    This assumes that ess.reflectometry is the primary piece of software
    used to reduce the data.
    """
    # Import here to break cycle __init__ -> io -> orso -> __init__
    from . import __version__

    return OrsoReduction(
        reduction.Reduction(
            software=reduction.Software(
                name="ess.reflectometry",
                version=str(__version__),
                platform=platform.system(),
            ),
            timestamp=datetime.now(tz=timezone.utc),
            creator=creator,
            corrections=[],
        )
    )


def build_orso_data_source(
    owner: OrsoOwner,
    sample: OrsoSample,
    experiment: OrsoExperiment,
    measurement: OrsoMeasurement,
) -> OrsoDataSource:
    """Assemble an ORSO DataSource."""
    return OrsoDataSource(
        data_source.DataSource(
            owner=owner,
            sample=sample,
            experiment=experiment,
            measurement=measurement,
        )
    )


def build_orso_instrument(events: ReducibleData[SampleRun]) -> OrsoInstrument:
    """Build ORSO instrument metadata from intermediate reduction results.

    This assumes specular reflection and sets the incident angle equal to the computed
    scattering angle.
    """
    return OrsoInstrument(
        data_source.InstrumentSettings(
            wavelength=orso_base.ValueRange(*_limits_of_coord(events, "wavelength")),
            incident_angle=orso_base.ValueRange(*_limits_of_coord(events, "theta")),
            polarization=None,  # TODO how can we determine this from the inputs?
        )
    )


def build_orso_iofq_dataset(
    iofq: ReflectivityOverQ,
    data_source: OrsoDataSource,
    reduction: OrsoReduction,
    corrections: OrsoCorrectionList,
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
    ds.info.reduction.corrections = list(corrections)
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
    min_ = coord.nanmin().value
    max_ = coord.nanmax().value
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


providers = (
    build_orso_data_source,
    build_orso_measurement,
    build_orso_reduction,
    parse_orso_experiment,
    parse_orso_owner,
    parse_orso_sample,
    orso_data_files,
    build_orso_instrument,
    build_orso_iofq_dataset,
)
