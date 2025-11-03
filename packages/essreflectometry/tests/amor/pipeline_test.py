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
from ess.amor import data
from ess.amor.types import ChopperPhase
from ess.reflectometry import orso
from ess.reflectometry.tools import batch_compute, scale_for_reflectivity_overlap
from ess.reflectometry.types import (
    BeamDivergenceLimits,
    DetectorRotation,
    Filename,
    ProtonCurrent,
    QBins,
    RawSampleRotation,
    ReducedReference,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    SampleRotation,
    SampleRotationOffset,
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
    pl = amor.AmorWorkflow()
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
    pl[Filename[ReferenceRun]] = amor.data.amor_run(614)

    pl[orso.OrsoCreator] = orso.OrsoCreator(
        fileio.base.Person(
            name="Max Mustermann",
            affiliation="European Spallation Source ERIC",
            contact="max.mustermann@ess.eu",
        )
    )
    return pl


def test_has_expected_coordinates(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_run(608)
    reflectivity_over_q = amor_pipeline.compute(ReflectivityOverQ)
    assert "Q" in reflectivity_over_q.coords
    assert "Q_resolution" in reflectivity_over_q.coords


def test_pipeline_no_gravity_correction(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_run(608)
    amor_pipeline[amor.types.GravityToggle] = False
    reflectivity_over_q = amor_pipeline.compute(ReflectivityOverQ)
    assert "Q" in reflectivity_over_q.coords
    assert "Q_resolution" in reflectivity_over_q.coords


def test_orso_pipeline(amor_pipeline: sciline.Pipeline):
    # The sample rotation value in the file is slightly off, so we set it manually
    amor_pipeline[SampleRotation[SampleRun]] = sc.scalar(0.85, unit="deg")
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_run(608)
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


def test_save_reduced_orso_file(output_folder: Path):
    from orsopy import fileio

    wf = amor.AmorWorkflow()
    wf[SampleSize[SampleRun]] = sc.scalar(10.0, unit="mm")
    wf[SampleSize[ReferenceRun]] = sc.scalar(10.0, unit="mm")
    wf[YIndexLimits] = sc.scalar(11), sc.scalar(41)
    wf[WavelengthBins] = sc.geomspace("wavelength", 3, 12.5, 2000, unit="angstrom")
    wf[ZIndexLimits] = sc.scalar(170), sc.scalar(266)
    wf[BeamDivergenceLimits] = (
        sc.scalar(-0.16, unit='deg'),
        sc.scalar(0.2, unit='deg'),
    )
    wf = with_filenames(
        wf, SampleRun, [data.amor_run(4079), data.amor_run(4080), data.amor_run(4081)]
    )
    wf[Filename[ReferenceRun]] = data.amor_run(4152)
    wf[QBins] = sc.geomspace(dim="Q", start=0.01, stop=0.06, num=201, unit="1/angstrom")

    r_of_q = wf.compute(ReflectivityOverQ)
    wf[ReflectivityOverQ] = r_of_q * scale_for_reflectivity_overlap(
        r_of_q,
        critical_edge_interval=(
            sc.scalar(0.01, unit='1/angstrom'),
            sc.scalar(0.014, unit='1/angstrom'),
        ),
    )
    wf[orso.OrsoCreator] = orso.OrsoCreator(
        fileio.base.Person(
            name="Max Mustermann",
            affiliation="European Spallation Source ERIC",
            contact="max.mustermann@ess.eu",
        )
    )
    fileio.orso.save_orso(
        datasets=[wf.compute(orso.OrsoIofQDataset)],
        fname=output_folder / 'amor_reduced_iofq.ort',
    )


def test_pipeline_can_compute_reflectivity_merging_events_from_multiple_runs(
    amor_pipeline: sciline.Pipeline,
):
    sample_runs = [
        amor.data.amor_run(608),
        amor.data.amor_run(609),
    ]
    wf = amor_pipeline.copy()
    wf[SampleRotationOffset[SampleRun]] = sc.scalar(0.05, unit="deg")
    pipeline = with_filenames(wf, SampleRun, sample_runs)
    result = pipeline.compute(ReflectivityOverQ)
    assert result.dims == ('Q',)


def test_pipeline_merging_events_result_unchanged(amor_pipeline: sciline.Pipeline):
    wf = amor_pipeline.copy()
    wf[SampleRotationOffset[SampleRun]] = sc.scalar(0.05, unit="deg")
    sample_runs = [
        amor.data.amor_run(608),
    ]
    pipeline = with_filenames(wf, SampleRun, sample_runs)
    result = pipeline.compute(ReflectivityOverQ).hist()
    sample_runs = [
        amor.data.amor_run(608),
        amor.data.amor_run(608),
    ]
    pipeline = with_filenames(wf, SampleRun, sample_runs)
    result2 = pipeline.compute(ReflectivityOverQ).hist()
    assert_allclose(
        2 * sc.values(result.data), sc.values(result2.data), rtol=sc.scalar(1e-6)
    )
    assert_allclose(
        2 * sc.variances(result.data), sc.variances(result2.data), rtol=sc.scalar(1e-6)
    )


def test_proton_current(amor_pipeline: sciline.Pipeline):
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_run(611)
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


def test_sample_rotation_offset(amor_pipeline: sciline.Pipeline):
    amor_pipeline[Filename[SampleRun]] = amor.data.amor_run(608)
    amor_pipeline[SampleRotationOffset[SampleRun]] = sc.scalar(1.0, unit='deg')
    mu, muoffset, muraw = amor_pipeline.compute(
        (
            SampleRotation[SampleRun],
            SampleRotationOffset[SampleRun],
            RawSampleRotation[SampleRun],
        )
    ).values()
    assert mu == muoffset.to(unit=muraw.unit) + muraw


@pytest.fixture
def pipeline_with_1632_reference(amor_pipeline):
    amor_pipeline[ChopperPhase[ReferenceRun]] = sc.scalar(7.5, unit='deg')
    amor_pipeline[ChopperPhase[SampleRun]] = sc.scalar(7.5, unit='deg')
    amor_pipeline[Filename[ReferenceRun]] = data.amor_run('1632')
    amor_pipeline[ReducedReference] = amor_pipeline.compute(ReducedReference)
    return amor_pipeline


def test_batch_compute_concatenates_event_lists(
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
    result = batch_compute(
        pl,
        {"": run},
        target=ReflectivityOverQ,
        scale_to_overlap=False,
    )[""]

    result2 = []
    for fname in run[Filename[SampleRun]]:
        pl.copy()
        pl[Filename[SampleRun]] = fname
        pl[QBins] = run[QBins]
        pl[DetectorRotation[SampleRun]] = run[DetectorRotation[SampleRun]]
        pl[SampleRotation[SampleRun]] = run[SampleRotation[SampleRun]]
        result2.append(pl.compute(ReflectivityOverQ).hist().data)

    assert_allclose(sum(result2), result.hist().data)
