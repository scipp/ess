# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""ORSO utilities for reflectometry."""
import platform
from datetime import datetime, timezone
from typing import NewType, Optional

import sciline
from orsopy.fileio import base as orso_base
from orsopy.fileio import data_source, reduction

from . import __version__
from .types import Reference, Run, Sample


class OrsoExperiment(
    sciline.Scope[Run, data_source.Experiment], data_source.Experiment
):
    """ORSO experiment for a run."""


class OrsoInstrument(
    sciline.Scope[Run, data_source.InstrumentSettings], data_source.InstrumentSettings
):
    """ORSO instrument settings for a run."""


class OrsoOwner(sciline.Scope[Run, orso_base.Person], orso_base.Person):
    """ORSO owner of a file."""


class OrsoReduction(sciline.Scope[Run, reduction.Reduction], reduction.Reduction):
    """ORSO measurement for a run."""


class OrsoSample(sciline.Scope[Run, data_source.Sample], data_source.Sample):
    """ORSO sample of a run."""


OrsoCreator = NewType('OrsoCreator', orso_base.Person)
"""ORSO creator, that is, the person who processed the data."""

OrsoDataSource = NewType('OrsoDataSource', data_source.DataSource)
"""ORSO data source."""

OrsoMeasurement = NewType('OrsoMeasurement', data_source.Measurement)
"""ORSO measurement."""


def build_orso_reduction(creator: Optional[OrsoCreator]) -> OrsoReduction:
    """Construct ORSO reduction data.

    This assumes that ess.reflectometry is the primary piece of software
    used to reduce the data.
    """
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
    owner: Optional[OrsoOwner[Sample]],
    sample: Optional[OrsoSample[Sample]],
    sample_experiment: Optional[OrsoExperiment[Sample]],
    reference_experiment: Optional[OrsoExperiment[Reference]],
    measurement: Optional[OrsoMeasurement],
) -> OrsoDataSource:
    """Judiciously assemble an ORSO DataSource.

    Makes some assumptions about how sample and reference runs should be merged,
    giving precedence to the sample run.
    """
    # We simply assume that the owner of the reference measurement
    # has no claim on this data.
    if (sample_experiment.facility != reference_experiment.facility) or (
        sample_experiment.instrument != reference_experiment.instrument
    ):
        raise ValueError(
            'The sample and reference experiments were done at different instruments'
        )

    return OrsoDataSource(
        data_source.DataSource(
            owner=owner,
            sample=sample,
            experiment=sample_experiment,
            measurement=measurement,
        )
    )


providers = (build_orso_reduction, build_orso_data_source)
