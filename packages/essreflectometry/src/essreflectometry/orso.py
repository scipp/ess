# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for reflectometry.

The Sciline providers and types in this module largely ignore the metadata
of reference runs and only use the metadata of the sample run.
"""

import os
import platform
from datetime import datetime, timezone
from typing import NewType, Optional

from dateutil.parser import parse as parse_datetime
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source, orso, reduction

from .types import Filename, RawData, Reference, Sample

OrsoCreator = NewType('OrsoCreator', orso_base.Person)
"""ORSO creator, that is, the person who processed the data."""

OrsoDataSource = NewType('OrsoDataSource', data_source.DataSource)
"""ORSO data source."""

OrsoExperiment = NewType('OrsoExperiment', data_source.Experiment)
"""ORSO experiment for the sample run."""

OrsoInstrument = NewType('OrsoInstrument', data_source.InstrumentSettings)
"""ORSO instrument settings for the sample run."""

OrsoIofQDataset = NewType('OrsoIofQDataset', orso.OrsoDataset)
"""ORSO dataset for reduced I-of-Q data."""

OrsoMeasurement = NewType('OrsoMeasurement', data_source.Measurement)
"""ORSO measurement."""

OrsoOwner = NewType('OrsoOwner', orso_base.Person)
"""ORSO owner of a measurement."""

OrsoReduction = NewType('OrsoReduction', reduction.Reduction)
"""ORSO data reduction metadata."""

OrsoSample = NewType('OrsoSample', data_source.Sample)
"""ORSO sample."""


def parse_orso_experiment(raw_data: RawData[Sample]) -> OrsoExperiment:
    """Parse ORSO experiment metadata from raw NeXus data."""
    return OrsoExperiment(
        data_source.Experiment(
            title=raw_data['title'],
            instrument=raw_data['instrument']['name'],
            facility=raw_data.get('facility'),
            start_date=parse_datetime(raw_data['start_time']),
            probe='neutron',
        )
    )


def parse_orso_owner(raw_data: RawData[Sample]) -> OrsoOwner:
    """Parse ORSO owner metadata from raw NeXus data."""
    return OrsoOwner(
        orso_base.Person(
            name=raw_data['user']['name'],
            contact=raw_data['user']['email'],
            affiliation=raw_data['user'].get('affiliation'),
        )
    )


def parse_orso_sample(raw_data: RawData[Sample]) -> OrsoSample:
    """Parse ORSO sample metadata from raw NeXus data."""
    if not raw_data.get('sample'):
        return OrsoSample(data_source.Sample.empty())
    raise NotImplementedError('NeXus sample parsing is not implemented')


def build_orso_measurement(
    sample_filename: Filename[Sample],
    reference_filename: Optional[Filename[Reference]],
    instrument: Optional[OrsoInstrument],
) -> OrsoMeasurement:
    """Assemble ORSO measurement metadata."""
    # TODO populate timestamp
    #      doesn't work with a local file because we need the timestamp of the original,
    #      SciCat can provide that
    if reference_filename:
        additional_files = [
            orso_base.File(
                file=os.path.basename(reference_filename), comment='supermirror'
            )
        ]
    else:
        additional_files = []
    return OrsoMeasurement(
        data_source.Measurement(
            instrument_settings=instrument,
            data_files=[orso_base.File(file=os.path.basename(sample_filename))],
            additional_files=additional_files,
        )
    )


def build_orso_reduction(creator: Optional[OrsoCreator]) -> OrsoReduction:
    """Construct ORSO reduction metadata.

    This assumes that ess.reflectometry is the primary piece of software
    used to reduce the data.
    """
    # Import here to break cycle __init__ -> io -> orso -> __init__
    from . import __version__

    return OrsoReduction(
        reduction.Reduction(
            software=reduction.Software(
                name='ess.reflectometry',
                version=str(__version__),
                platform=platform.system(),
            ),
            timestamp=datetime.now(tz=timezone.utc),
            creator=creator,
            corrections=[],
        )
    )


def build_orso_data_source(
    owner: Optional[OrsoOwner],
    sample: Optional[OrsoSample],
    experiment: Optional[OrsoExperiment],
    measurement: Optional[OrsoMeasurement],
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


providers = (
    build_orso_data_source,
    build_orso_measurement,
    build_orso_reduction,
    parse_orso_experiment,
    parse_orso_owner,
    parse_orso_sample,
)
