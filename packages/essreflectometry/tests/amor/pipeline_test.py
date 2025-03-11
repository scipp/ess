# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from pathlib import Path

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
    ProtonCurrent,
    QBins,
    ReducibleData,
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

# The files used in the AMOR reduction workflow have some scippnexus warnings
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Invalid transformation, .*missing attribute 'vector':UserWarning",
)


@pytest.fixture
def amor_pipeline() -> sciline.Pipeline:
    pl = sciline.Pipeline(providers=amor.providers, params=amor.default_parameters())
    pl[SampleSize[SampleRun]] = sc.scalar(10.0, unit="mm")
    pl[SampleSize[ReferenceRun]] = sc.scalar(10.0, unit="mm")

    pl[WavelengthBins] = sc.geomspace("wavelength", 2.8, 12, 300, unit="angstrom")
    pl[YIndexLimits] = sc.scalar(11), sc.scalar(41)
    pl[ZIndexLimits] = sc.scalar(80), sc.scalar(370)
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
def test_has_expected_coordinates(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    reflectivity_over_q = amor_pipeline.compute(ReflectivityOverQ)
    assert "Q" in reflectivity_over_q.coords
    assert "Q_resolution" in reflectivity_over_q.coords


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_pipeline_no_gravity_correction(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    amor_pipeline[amor.types.GravityToggle] = False
    reflectivity_over_q = amor_pipeline.compute(ReflectivityOverQ)
    assert "Q" in reflectivity_over_q.coords
    assert "Q_resolution" in reflectivity_over_q.coords


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_orso_pipeline(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    res = amor_pipeline.compute(orso.OrsoIofQDataset)
    assert res.info.data_source.experiment.instrument == "Amor"
    assert res.info.reduction.software.name == "ess.reflectometry"
    assert res.info.reduction.corrections == [
        "chopper ToF correction",
        "footprint correction",
        "supermirror calibration",
    ]
    assert res.data.ndim == 2
    assert res.data.shape[1] == 4
    assert np.all(res.data[:, 1] >= 0)
    assert np.isfinite(res.data).all()


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_save_reduced_orso_file(amor_pipeline: sciline.Pipeline, output_folder: Path):
    from orsopy import fileio

    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(608)
    res = amor_pipeline.compute(orso.OrsoIofQDataset)
    fileio.orso.save_orso(datasets=[res], fname=output_folder / 'amor_reduced_iofq.ort')


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


@pytest.mark.filterwarnings("ignore:Failed to convert .* into a transformation")
@pytest.mark.filterwarnings("ignore:Invalid transformation, missing attribute")
def test_proton_current(amor_pipeline: sciline.Pipeline):
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_sample_run(611)
    da_without_proton_current = amor_pipeline.compute(ReducibleData[SampleRun])

    proton_current = [1, 2, 0.1]
    timestamps = [1699883542349602112, 1699883542349602112, 1699886864071691036]
    amor_pipeline[ProtonCurrent[SampleRun]] = sc.DataArray(
        sc.array(dims=['time'], values=proton_current),
        coords={
            'time': sc.array(
                dims=['time'],
                values=timestamps,
                dtype='datetime64',
                unit='ns',
            )
        },
    )
    da_with_proton_current = amor_pipeline.compute(ReducibleData[SampleRun])

    assert "proton_current" in da_with_proton_current.bins.coords
    assert "proton_current_too_low" in da_with_proton_current.bins.masks
    assert da_with_proton_current.bins.masks["proton_current_too_low"].any()
    assert not da_with_proton_current.bins.masks["proton_current_too_low"].all()

    assert "proton_current" not in da_without_proton_current.bins.coords
    assert "proton_current_too_low" not in da_without_proton_current.bins.masks

    t = (
        da_with_proton_current.bins.constituents['data']
        .coords['event_time_zero'][0]
        .value.astype('uint64')
    )
    w_with = da_with_proton_current.bins.constituents['data'].data[0].value
    w_without = da_without_proton_current.bins.constituents['data'].data[0].value
    np.testing.assert_allclose(
        proton_current[np.searchsorted(timestamps, t) - 1], w_without / w_with
    )
