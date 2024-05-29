# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for reflectometry.

The Sciline providers and types in this module largely ignore the metadata
of reference runs and only use the metadata of the sample run.
"""

import graphlib
import os
import platform
from datetime import datetime, timezone
from typing import Any, NewType, Optional

from dateutil.parser import parse as parse_datetime
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source, orso, reduction

from .load import load_nx
from .supermirror import SupermirrorReflectivityCorrection
from .types import (
    ChopperCorrectedTofEvents,
    FilePath,
    FootprintCorrectedData,
    Reference,
    Sample,
)

try:
    from sciline.task_graph import TaskGraph
except ModuleNotFoundError:
    TaskGraph = Any


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


def parse_orso_experiment(filepath: FilePath[Sample]) -> OrsoExperiment:
    """Parse ORSO experiment metadata from raw NeXus data."""
    title, instrument_name, facility, start_time = load_nx(
        filepath,
        'NXentry/title',
        'NXentry/NXinstrument/name',
        'NXentry/facility',
        'NXentry/start_time',
    )
    return OrsoExperiment(
        data_source.Experiment(
            title=title,
            instrument=instrument_name,
            facility=facility,
            start_date=parse_datetime(start_time),
            probe='neutron',
        )
    )


def parse_orso_owner(filepath: FilePath[Sample]) -> OrsoOwner:
    """Parse ORSO owner metadata from raw NeXus data."""
    (user,) = load_nx(filepath, 'NXentry/NXuser')
    return OrsoOwner(
        orso_base.Person(
            name=user['name'],
            contact=user['email'],
            affiliation=user.get('affiliation'),
        )
    )


def parse_orso_sample(filepath: FilePath[Sample]) -> OrsoSample:
    """Parse ORSO sample metadata from raw NeXus data."""
    (sample,) = load_nx(filepath, 'NXentry/NXsample')
    if not sample:
        return OrsoSample(data_source.Sample.empty())
    return OrsoSample(
        data_source.Sample(
            name=sample['name'],
            model=data_source.SampleModel(
                stack=sample['model'],
            ),
        )
    )


def build_orso_measurement(
    sample_filepath: FilePath[Sample],
    reference_filepath: Optional[FilePath[Reference]],
    instrument: Optional[OrsoInstrument],
) -> OrsoMeasurement:
    """Assemble ORSO measurement metadata."""
    # TODO populate timestamp
    #      doesn't work with a local file because we need the timestamp of the original,
    #      SciCat can provide that
    if reference_filepath:
        additional_files = [
            orso_base.File(
                file=os.path.basename(reference_filepath), comment='supermirror'
            )
        ]
    else:
        additional_files = []
    return OrsoMeasurement(
        data_source.Measurement(
            instrument_settings=instrument,
            data_files=[orso_base.File(file=os.path.basename(sample_filepath))],
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


_CORRECTIONS_BY_GRAPH_KEY = {
    ChopperCorrectedTofEvents[Sample]: 'chopper ToF correction',
    FootprintCorrectedData[Sample]: 'footprint correction',
    SupermirrorReflectivityCorrection: 'supermirror calibration',
}


def find_corrections(task_graph: TaskGraph) -> list[str]:
    """Determine the list of corrections for ORSO from a task graph.

    Checks for known keys in the graph that correspond to corrections
    that should be tracked in an ORSO output dataset.
    Bear in mind that this exclusively checks the types used as keys in a task graph,
    it cannot detect other corrections that are performed within providers
    or outside the graph.

    Parameters
    ----------
    :
        task_graph:
            The task graph used to produce output data.

    Returns
    -------
    :
        List of corrections in the order they are applied in.
    """
    toposort = graphlib.TopologicalSorter(
        {
            key: tuple(provider.arg_spec.keys())
            for key, provider in task_graph._graph.items()
        }
    )
    return [
        c
        for key in toposort.static_order()
        if (c := _CORRECTIONS_BY_GRAPH_KEY.get(key, None)) is not None
    ]


providers = (
    build_orso_data_source,
    build_orso_measurement,
    build_orso_reduction,
    parse_orso_experiment,
    parse_orso_owner,
    parse_orso_sample,
)
