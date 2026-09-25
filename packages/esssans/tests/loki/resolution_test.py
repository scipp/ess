# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import ess.loki.data  # noqa: F401
import numpy as np
import pytest
import sciline
import scipp as sc
from ess import loki, sans
from ess.sans.resolution import q_resolution
from ess.sans.types import (
    BeamCenter,
    CollimationLength,
    Denominator,
    DetectorPixelSize,
    DetectorQVariance,
    DimsToKeep,
    Filename,
    IntensityQ,
    LookupTableFilename,
    MonitorTerm,
    NeXusDetectorName,
    QDetector,
    QResolution,
    ReducedQ,
    ResolutionFirstMoment,
    ResolutionSecondMoment,
    SampleApertureRadius,
    SampleRun,
    SourceApertureRadius,
    SourceWavelengthSpread,
    WavelengthBands,
    WavelengthBins,
)
from scipp.testing import assert_allclose


def _set_resolution_parameters(wf: sciline.Pipeline) -> None:
    wf[SourceApertureRadius] = sc.scalar(15.0, unit='mm')
    wf[SampleApertureRadius] = sc.scalar(4.0, unit='mm')
    wf[CollimationLength] = sc.scalar(4.8, unit='m')
    wf[DetectorPixelSize] = sc.scalar(8.0, unit='mm')


@pytest.fixture
def workflow(larmor_workflow) -> sciline.Pipeline:
    wf = larmor_workflow(no_masks=False)
    wf = sans.with_pixel_mask_filenames(wf, loki.data.loki_tutorial_mask_filenames())
    wf[BeamCenter] = sc.vector([-0.0291487, -0.0181614, 0], unit='m')
    _set_resolution_parameters(wf)
    # The Larmor data is not converted to wavelength with a lookup table, so we use a
    # constant relative wavelength spread.
    wf[SourceWavelengthSpread[SampleRun]] = 0.02 * sc.midpoints(
        wf.compute(WavelengthBins)
    )
    return wf


def test_q_resolution_equals_n_weighted_second_moment(
    workflow: sciline.Pipeline,
) -> None:
    results = workflow.compute(
        [
            QResolution[SampleRun],
            QDetector[SampleRun, Denominator],
            DetectorQVariance[SampleRun],
            MonitorTerm[SampleRun],
        ]
    )
    resolution = results[QResolution[SampleRun]]
    detector_term = results[QDetector[SampleRun, Denominator]]
    variance = results[DetectorQVariance[SampleRun]]

    # Brute force over the dense (pixel, wavelength) arrays
    dims = detector_term.dims
    unmasked = ~sc.reduce(list(detector_term.masks.values())).any()
    n = sc.values(detector_term.data) * sc.values(
        results[MonitorTerm[SampleRun]].data
    )
    n = sc.where(unmasked, n, sc.zeros_like(n)).transpose(dims).values.ravel()
    q = detector_term.coords['Q'].transpose(dims)
    second = (variance.transpose(dims) + q**2).values.ravel()
    q = q.values.ravel()
    edges = resolution.coords['Q'].values
    expected_mean = []
    expected_variance = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (q >= lo) & (q < hi)
        w = n[sel] / n[sel].sum()
        mean = np.sum(w * q[sel])
        expected_mean.append(mean)
        expected_variance.append(np.sum(w * second[sel]) - mean**2)

    np.testing.assert_allclose(
        resolution.coords['Q_mean'].values, expected_mean, rtol=1e-12
    )
    np.testing.assert_allclose(
        resolution.values**2, expected_variance, rtol=1e-9
    )


def test_q_resolution_of_merged_runs_equals_resolution_from_summed_moments(
    workflow: sciline.Pipeline,
) -> None:
    runs = [
        loki.data.loki_tutorial_sample_run_60250(),
        loki.data.loki_tutorial_sample_run_60339(),
    ]
    parts = (Denominator, ResolutionFirstMoment, ResolutionSecondMoment)
    per_run = []
    for run in runs:
        workflow[Filename[SampleRun]] = run
        per_run.append([workflow.compute(ReducedQ[SampleRun, part]) for part in parts])
    expected = q_resolution(*(a + b for a, b in zip(*per_run, strict=True)))

    merged = sans.with_sample_runs(workflow, runs=runs)
    result = merged.compute(QResolution[SampleRun])
    assert_allclose(result.data, expected.data)
    assert_allclose(result.coords['Q_mean'], expected.coords['Q_mean'])


@pytest.mark.parametrize(
    ('key', 'value', 'dim'),
    [(WavelengthBands, None, 'band'), (DimsToKeep, ['layer'], 'layer')],
)
def test_q_resolution_has_same_dims_as_iofq(
    workflow: sciline.Pipeline, key: type, value: object, dim: str
) -> None:
    if key is WavelengthBands:
        edges = workflow.compute(WavelengthBins)
        value = sc.linspace('wavelength', edges.min(), edges.max(), 4)
    workflow[key] = value
    results = workflow.compute([QResolution[SampleRun], IntensityQ[SampleRun]])
    resolution = results[QResolution[SampleRun]]
    iofq = results[IntensityQ[SampleRun]]
    assert dim in resolution.dims
    assert resolution.dims == iofq.dims
    assert resolution.coords['Q_mean'].dims == iofq.dims
    assert sc.identical(resolution.coords['Q'], iofq.coords['Q'])


def test_loki_q_resolution_uses_wavelength_spread_from_lookup_table(
    loki_workflow,
) -> None:
    wf = loki_workflow()
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = 'loki_detector_0'
    wf[LookupTableFilename] = loki.data.loki_lookup_table_no_choppers()
    _set_resolution_parameters(wf)

    spread = wf.compute(SourceWavelengthSpread[SampleRun])
    assert sc.isfinite(spread).all()
    assert (spread > sc.scalar(0.0, unit=spread.unit)).all()
    # The test file contains no real data, so we only check that the result exists
    assert wf.compute(QResolution[SampleRun]).dims == ('Q',)
