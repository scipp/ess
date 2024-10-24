# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F403, F405

import numpy as np
import pytest
import sciline
import scipp as sc
from orsopy import fileio
from scipp.testing import assert_allclose

from ess import amor
from ess.amor import data  # noqa: F401
from ess.reflectometry import orso
from ess.reflectometry.types import (
    Filename,
    QBins,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRotation,
    SampleRun,
    SampleSize,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from ess.reflectometry.workflow import with_filenames


@pytest.fixture
def amor_pipeline() -> sciline.Pipeline:
    pl = sciline.Pipeline(providers=amor.providers, params=amor.default_parameters())
    pl[SampleSize[SampleRun]] = sc.scalar(10.0, unit="mm")
    pl[SampleSize[ReferenceRun]] = sc.scalar(10.0, unit="mm")

    pl[WavelengthBins] = sc.geomspace("wavelength", 2.8, 12, 300, unit="angstrom")
    pl[YIndexLimits] = sc.scalar(11, unit=None), sc.scalar(41, unit=None)
    pl[ZIndexLimits] = sc.scalar(80, unit=None), sc.scalar(370, unit=None)
    pl[QBins] = sc.geomspace(
        dim="Q", start=0.005, stop=0.115, num=391, unit="1/angstrom"
    )

    # The sample rotation value in the file is slightly off, so we set it manually
    pl[SampleRotation[ReferenceRun]] = sc.scalar(0.65, unit="deg")
    pl[Filename[ReferenceRun]] = amor.data.amor_reference_run()

    pl[orso.OrsoCreator] = orso.OrsoCreator(
        fileio.base.Person(
            name="Max Mustermann",
            affiliation="European Spallation Source ERIC",
            contact="max.mustermann@ess.eu",
        )
    )
    return pl


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_run_data_pipeline(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    res = amor_pipeline.compute(ReflectivityOverQ)
    assert "Q" in res.coords
    assert "Q_resolution" in res.coords


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_run_full_pipeline(amor_pipeline: sciline.Pipeline):
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    # Make the Q range cover a larger interval than the experiment is sensitive to.
    # This let's us test the non-covered regions are filtered from the ORSO data.
    amor_pipeline[QBins] = sc.geomspace(
        dim="Q", start=0.005, stop=0.15, num=391, unit="1/angstrom"
    )
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    res = amor_pipeline.compute(orso.OrsoIofQDataset)
    assert res.info.data_source.experiment.instrument == "Amor"
    assert res.info.reduction.software.name == "ess.reflectometry"
    assert res.data.ndim == 2
    assert res.data.shape[1] == 4
    assert np.all(res.data[:, 1] >= 0)
    assert np.isfinite(res.data).all()


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_pipeline_can_compute_reflectivity_merging_events_from_multiple_runs(
    amor_pipeline: sciline.Pipeline,
):
    sample_runs = [
        amor.data.amor_sample_run(608),
        amor.data.amor_sample_run(609),
    ]
    pipeline = with_filenames(amor_pipeline, SampleRun, sample_runs)
    pipeline[SampleRotation[SampleRun]] = pipeline.compute(
        SampleRotation[SampleRun]
    ) + sc.scalar(0.05, unit="deg")
    result = pipeline.compute(ReflectivityOverQ)
    assert result.dims == ('Q',)


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_pipeline_merging_events_result_unchanged(amor_pipeline: sciline.Pipeline):
    sample_runs = [
        amor.data.amor_sample_run(608),
    ]
    pipeline = with_filenames(amor_pipeline, SampleRun, sample_runs)
    pipeline[SampleRotation[SampleRun]] = pipeline.compute(
        SampleRotation[SampleRun]
    ) + sc.scalar(0.05, unit="deg")
    result = pipeline.compute(ReflectivityOverQ).hist()
    sample_runs = [
        amor.data.amor_sample_run(608),
        amor.data.amor_sample_run(608),
    ]
    pipeline = with_filenames(amor_pipeline, SampleRun, sample_runs)
    pipeline[SampleRotation[SampleRun]] = pipeline.compute(
        SampleRotation[SampleRun]
    ) + sc.scalar(0.05, unit="deg")
    result2 = pipeline.compute(ReflectivityOverQ).hist()
    assert_allclose(
        2 * sc.values(result.data), sc.values(result2.data), rtol=sc.scalar(1e-6)
    )
    assert_allclose(
        2 * sc.variances(result.data), sc.variances(result2.data), rtol=sc.scalar(1e-6)
    )


def test_find_corrections(amor_pipeline: sciline.Pipeline):
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    graph = amor_pipeline.get(orso.OrsoIofQDataset)
    # In topological order
    assert orso.find_corrections(graph) == [
        "chopper ToF correction",
        "footprint correction",
        "supermirror calibration",
    ]
