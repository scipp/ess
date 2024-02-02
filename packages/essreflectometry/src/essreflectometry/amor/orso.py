# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for Amor."""
import os
from typing import Optional

from dateutil.parser import parse as parse_datetime
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source

from ..orso import (
    OrsoExperiment,
    OrsoInstrument,
    OrsoMeasurement,
    OrsoOwner,
    OrsoSample,
)
from ..types import Filename, RawData, Reference, Run, Sample


def parse_orso_experiment(raw_data: RawData[Run]) -> OrsoExperiment[Run]:
    """Parse ORSO experiment data from raw Amor NeXus data."""
    return OrsoExperiment(
        data_source.Experiment(
            title=raw_data['title'],
            instrument=raw_data['name'],
            facility=raw_data['facility'],
            start_date=parse_datetime(raw_data['start_time']),
            probe='neutron',
        )
    )


def parse_orso_owner(raw_data: RawData[Run]) -> OrsoOwner[Run]:
    """Parse ORSO owner data from raw Amor NeXus data."""
    return OrsoOwner(
        orso_base.Person(
            name=raw_data['user']['name'],
            contact=raw_data['user']['email'],
            affiliation=None,
        )
    )


def parse_orso_sample(raw_data: RawData[Run]) -> OrsoSample[Run]:
    """Parse ORSO sample data from raw Amor NeXus data."""
    if not raw_data.get('sample'):
        return OrsoSample(data_source.Sample.empty())
    raise NotImplementedError('Amor NsXus sample parsing is not implemented')


def build_orso_measurement(
    sample_filename: Filename[Sample],
    reference_filename: Filename[Reference],
    instrument: Optional[OrsoInstrument],
) -> OrsoMeasurement:
    """Assemble ORSO measurement data."""
    # TODO populate timestamp
    #      doesn't work with a local file because we need the timestamp of the original,
    #      SciCat can provide that
    return OrsoMeasurement(
        data_source.Measurement(
            instrument_settings=instrument,
            data_files=[orso_base.File(file=os.path.basename(sample_filename))],
            additional_files=[
                orso_base.File(
                    file=os.path.basename(reference_filename), comment='supermirror'
                )
            ],
        )
    )


providers = (
    parse_orso_experiment,
    build_orso_measurement,
    parse_orso_owner,
    parse_orso_sample,
)
