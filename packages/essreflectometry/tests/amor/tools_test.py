# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc
from scipp.testing import assert_allclose

from amor.pipeline_test import amor_pipeline  # noqa: F401
from ess.amor import data
from ess.amor.types import ChopperPhase
from ess.reflectometry.tools import from_measurements
from ess.reflectometry.types import (
    DetectorRotation,
    Filename,
    QBins,
    ReducedReference,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRotation,
    SampleRun,
)

# The files used in the AMOR reduction workflow have some scippnexus warnings
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Invalid transformation, .*missing attribute 'vector':UserWarning",
)


@pytest.fixture
def pipeline_with_1632_reference(amor_pipeline):  # noqa: F811
    amor_pipeline[ChopperPhase[ReferenceRun]] = sc.scalar(7.5, unit='deg')
    amor_pipeline[ChopperPhase[SampleRun]] = sc.scalar(7.5, unit='deg')
    amor_pipeline[Filename[ReferenceRun]] = data.amor_run('1632')
    amor_pipeline[ReducedReference] = amor_pipeline.compute(ReducedReference)
    return amor_pipeline


@pytestmark
def test_from_measurements_tool_concatenates_event_lists(
    pipeline_with_1632_reference: sciline.Pipeline,
):
    pl = pipeline_with_1632_reference

    run = {
        Filename[SampleRun]: list(map(data.amor_run, (1636, 1639, 1641))),
        QBins: sc.geomspace(
            dim='Q', start=0.062, stop=0.18, num=391, unit='1/angstrom'
        ),
        DetectorRotation[SampleRun]: sc.scalar(0.140167, unit='rad'),
        SampleRotation[SampleRun]: sc.scalar(0.0680678, unit='rad'),
    }
    results = from_measurements(
        pl,
        [run],
        target=ReflectivityOverQ,
        scale_to_overlap=False,
    )

    results2 = []
    for fname in run[Filename[SampleRun]]:
        pl.copy()
        pl[Filename[SampleRun]] = fname
        pl[QBins] = run[QBins]
        pl[DetectorRotation[SampleRun]] = run[DetectorRotation[SampleRun]]
        pl[SampleRotation[SampleRun]] = run[SampleRotation[SampleRun]]
        results2.append(pl.compute(ReflectivityOverQ).hist().data)

    assert_allclose(sum(results2), results[0].hist().data)
