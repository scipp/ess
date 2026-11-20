# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from typing import NewType

import numpy as np
import pytest
import sciline
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.reduce import unwrap
from ess.reduce.nexus.types import (
    DiskChoppers,
    EmptyDetector,
    EmptyMonitor,
    NeutronSourcePosition,
    NeXusData,
    NeXusDetectorName,
    NeXusName,
    Position,
    RawDetector,
    RawMonitor,
    SampleRun,
)
from ess.reduce.unwrap import (
    FrameUnwrapBackend,
    GenericUnwrapWorkflow,
    fakes,
    simulate_chopper_cascade_using_tof,
)
from ess.reduce.unwrap.types import KeepEventTimeOffset

sl = pytest.importorskip("sciline")

Monitor0 = NewType("Monitor0", int)
Monitor1 = NewType("Monitor1", int)


def test_GenericUnwrapWorkflow_defaults_to_numba_backend():
    wf = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[])

    assert wf.compute(FrameUnwrapBackend) == FrameUnwrapBackend.numba


def test_monitor_wavelength_bins_are_independent_and_default_to_none():
    wf = GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[Monitor0, Monitor1]
    )
    assert wf.compute(unwrap.MonitorWavelengthBins[Monitor0]) is None
    assert wf.compute(unwrap.MonitorWavelengthBins[Monitor1]) is None

    edges = sc.linspace('wavelength', 1.0, 5.0, 3, unit='angstrom')
    wf[unwrap.MonitorWavelengthBins[Monitor0]] = edges
    assert_identical(wf.compute(unwrap.MonitorWavelengthBins[Monitor0]), edges)
    assert wf.compute(unwrap.MonitorWavelengthBins[Monitor1]) is None


@pytest.mark.parametrize('bin_dim', ['wavelength', 'incident_wavelength', 'wavelenght'])
def test_monitor_wavelength_bins_rebin_histograms(bin_dim):
    wf = _make_workflow(wavelength_from='analytical')
    target = unwrap.WavelengthMonitor[SampleRun, Monitor0]
    original = wf.compute(target)
    mid = original.sizes['wavelength'] // 2
    wavelengths = original.coords['wavelength']
    edges = sc.concat([wavelengths[0], wavelengths[mid], wavelengths[-1]], bin_dim)
    expected = sc.concat(
        [
            original['wavelength', :mid].sum('wavelength').data,
            original['wavelength', mid:].sum('wavelength').data,
        ],
        'wavelength',
    ).transpose(original.dims)

    wf[unwrap.MonitorWavelengthBins[Monitor0]] = edges
    result = wf.compute(target)

    sc.testing.assert_allclose(result.data, expected)
    assert_identical(
        result.coords['wavelength'], edges.rename_dims({bin_dim: 'wavelength'})
    )
    assert edges.dim == bin_dim


@pytest.mark.parametrize('bin_dim', ['wavelength', 'incident_wavelength', 'wavelenght'])
def test_monitor_wavelength_bins_histogram_events(bin_dim):
    wf = _make_workflow(wavelength_from='analytical')
    wf[DiskChoppers[SampleRun]] = {}
    events = sc.DataArray(
        sc.ones(dims=['event'], shape=[4], unit='counts', with_variances=True),
        coords={
            'event_time_offset': sc.array(
                dims=['event'], values=[15.0, 15.0, 30.0, 45.0], unit='ms'
            )
        },
    )
    wf[RawMonitor[SampleRun, Monitor0]] = sc.DataArray(
        sc.bins(data=events, dim='event', begin=sc.index(0), end=sc.index(4))
    )
    target = unwrap.WavelengthMonitor[SampleRun, Monitor0]
    original = wf.compute(target)
    assert original.is_binned
    wavelengths = original.bins.constituents['data'].coords['wavelength']
    edges = sc.concat(
        [
            sc.scalar(0.0, unit='angstrom'),
            (wavelengths[1] + wavelengths[2]) / 2,
            (wavelengths[2] + wavelengths[3]) / 2,
            sc.scalar(15.0, unit='angstrom'),
        ],
        bin_dim,
    )
    wf[unwrap.MonitorWavelengthBins[Monitor0]] = edges

    result = wf.compute(target)

    assert not result.is_binned
    assert_identical(
        result.data,
        sc.array(
            dims=['wavelength'],
            values=[2.0, 1.0, 1.0],
            variances=[2.0, 1.0, 1.0],
            unit='counts',
        ),
    )
    assert_identical(
        result.coords['wavelength'], edges.rename_dims({bin_dim: 'wavelength'})
    )
    assert edges.dim == bin_dim


def _make_workflow(
    wavelength_from,
    *,
    backend: FrameUnwrapBackend = FrameUnwrapBackend.numba,
    keep_event_time_offset=False,
) -> sciline.Pipeline:
    sizes = {'detector_number': 10}
    detector_geometry = sc.DataArray(
        data=sc.ones(sizes=sizes),
        coords={
            "position": sc.spatial.as_vectors(
                sc.zeros(sizes=sizes, unit='m'),
                sc.zeros(sizes=sizes, unit='m'),
                sc.linspace("detector_number", 79, 81, 10, unit='m'),
            ),
            "detector_number": sc.array(
                dims=["detector_number"], values=np.arange(10), unit=None
            ),
        },
    )

    detector_events = sc.DataArray(
        data=sc.ones(dims=["event"], shape=[1000]),
        coords={
            "event_time_offset": sc.linspace(
                "event", 0.0, 1000.0 / 14, num=1000, unit="ms"
            ).to(unit="ns"),
            "event_id": sc.array(
                dims=["event"], values=np.arange(1000) % 10, unit=None
            ),
        },
    )
    detector_data = sc.DataArray(
        sc.bins(
            begin=sc.array(dims=["pulse"], values=[0], unit=None),
            data=detector_events,
            dim="event",
        )
    )

    monitor_geometry = sc.DataArray(
        data=sc.scalar(0.0), coords={"position": sc.vector([0, 0, 75], unit='m')}
    )

    monitor_data = sc.DataArray(
        data=sc.ones(sizes={'time': 10})
        * sc.arange("frame_time", 0, 300, unit='counts'),
        coords={
            "time": sc.array(dims=["time"], values=np.arange(10), unit=None),
            "frame_time": sc.linspace("frame_time", 0, 71, 301, unit='ms'),
        },
    )

    wf = GenericUnwrapWorkflow(
        run_types=[SampleRun],
        monitor_types=[Monitor0],
        wavelength_from=wavelength_from,
    )
    wf[FrameUnwrapBackend] = backend
    wf[NeXusDetectorName] = "detector"
    wf[NeXusName[Monitor0]] = "monitor"
    wf[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': np.inf,
        'monitor': np.inf,
    }
    wf[EmptyDetector[SampleRun]] = detector_geometry
    wf[NeXusData[snx.NXdetector, SampleRun]] = detector_data
    wf[EmptyMonitor[SampleRun, Monitor0]] = monitor_geometry
    wf[NeXusData[Monitor0, SampleRun]] = monitor_data
    wf[Position[snx.NXsample, SampleRun]] = sc.vector([0, 0, 77], unit='m')
    wf[NeutronSourcePosition[SampleRun]] = fakes.source_position()
    wf[DiskChoppers[SampleRun]] = fakes.psc_choppers()
    wf[KeepEventTimeOffset] = keep_event_time_offset

    return wf


@pytest.fixture(scope="module")
def simulation_results_psc_choppers():
    return simulate_chopper_cascade_using_tof(
        choppers=fakes.psc_choppers(),
        source_position=fakes.source_position(),
        neutrons=1e6,
        pulse_stride=1,
        seed=333,
        facility="ess",
        wmin=None,
        wmax=None,
    )


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("backend", ["numba", "scipy"])
def test_GenericUnwrapWorkflow_computes_wavelength(
    wavelength_from, detector_or_monitor, backend, simulation_results_psc_choppers
):
    wf = _make_workflow(wavelength_from=wavelength_from, backend=backend)

    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    if detector_or_monitor == "monitor":
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, Monitor0])
        assert 'wavelength' in wavs.coords
    else:
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
        assert 'wavelength' in wavs.bins.coords


@pytest.mark.parametrize("keep_event_time_offset", [False, True])
def test_GenericUnwrapWorkflow_optionally_keeps_event_time_offset(
    keep_event_time_offset,
):
    wf = _make_workflow(
        wavelength_from="analytical",
        keep_event_time_offset=keep_event_time_offset,
    )

    raw = wf.compute(RawDetector[SampleRun])
    wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])

    if keep_event_time_offset:
        assert_identical(
            wavs.bins.coords['event_time_offset'],
            raw.bins.coords['event_time_offset'],
        )
    else:
        assert 'event_time_offset' not in wavs.bins.coords


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
def test_GenericUnwrapWorkflow_makes_different_luts_for_detector_and_monitor(
    wavelength_from, simulation_results_psc_choppers
):
    wf = _make_workflow(wavelength_from=wavelength_from)
    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    det_table = wf.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])
    mon_table = wf.compute(unwrap.LookupTable[SampleRun, Monitor0])

    assert det_table.array.sizes['distance'] != mon_table.array.sizes['distance']
    assert (
        det_table.array.sizes['event_time_offset']
        == mon_table.array.sizes['event_time_offset']
    )


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
def test_GenericUnwrapWorkflow_with_lut_from_file(
    wavelength_from, tmp_path: pytest.TempPathFactory, simulation_results_psc_choppers
):
    wf = _make_workflow(wavelength_from=wavelength_from)

    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    wf[unwrap.LtotalRange[SampleRun, snx.NXdetector]] = (
        sc.scalar(75.0, unit="m"),
        sc.scalar(85.0, unit="m"),
    )
    lut = wf.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])
    lut.save_hdf5(filename=tmp_path / "lut.h5")

    wf_from_file = _make_workflow(wavelength_from="file")
    wf_from_file[unwrap.LookupTableFilename[SampleRun, snx.NXdetector]] = (
        tmp_path / "lut.h5"
    ).as_posix()

    loaded_lut = wf_from_file.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])
    assert_identical(lut.array, loaded_lut.array)
    assert_identical(lut.pulse_period, loaded_lut.pulse_period)
    assert lut.pulse_stride == loaded_lut.pulse_stride
    assert_identical(lut.distance_resolution, loaded_lut.distance_resolution)
    assert_identical(lut.time_resolution, loaded_lut.time_resolution)
    assert_identical(lut.choppers, loaded_lut.choppers)

    detector = wf_from_file.compute(unwrap.WavelengthDetector[SampleRun])
    assert 'wavelength' in detector.bins.coords


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
def test_GenericUnwrapWorkflow_with_lut_from_file_old_format(
    wavelength_from, tmp_path: pytest.TempPathFactory, simulation_results_psc_choppers
):
    wf = _make_workflow(wavelength_from=wavelength_from)

    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    wf[unwrap.LtotalRange[SampleRun, snx.NXdetector]] = (
        sc.scalar(75.0, unit="m"),
        sc.scalar(85.0, unit="m"),
    )
    lut = wf.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])
    old_lut = sc.DataArray(
        data=lut.array.data,
        coords={
            "distance": lut.array.coords["distance"],
            "event_time_offset": lut.array.coords["event_time_offset"],
            "pulse_period": lut.pulse_period,
            "pulse_stride": sc.scalar(lut.pulse_stride, unit=None),
            "distance_resolution": lut.distance_resolution,
            "time_resolution": lut.time_resolution,
        },
    )
    old_lut.save_hdf5(filename=tmp_path / "lut.h5")

    wf_from_file = _make_workflow(wavelength_from="file")
    wf_from_file[unwrap.LookupTableFilename[SampleRun, snx.NXdetector]] = (
        tmp_path / "lut.h5"
    ).as_posix()
    loaded_lut = wf_from_file.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])
    assert_identical(lut.array, loaded_lut.array)
    assert_identical(lut.pulse_period, loaded_lut.pulse_period)
    assert lut.pulse_stride == loaded_lut.pulse_stride
    assert_identical(lut.distance_resolution, loaded_lut.distance_resolution)
    assert_identical(lut.time_resolution, loaded_lut.time_resolution)
    assert loaded_lut.choppers is None  # No chopper info in old format

    detector = wf_from_file.compute(unwrap.WavelengthDetector[SampleRun])
    assert 'wavelength' in detector.bins.coords
