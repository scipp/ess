# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
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
    FrameMonitor0,
    NeXusData,
    NeXusDetectorName,
    NeXusName,
    Position,
    SampleRun,
)
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    fakes,
    simulate_chopper_cascade_using_tof,
)

sl = pytest.importorskip("sciline")


def _make_workflow(wavelength_from) -> sciline.Pipeline:
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
        monitor_types=[FrameMonitor0],
        wavelength_from=wavelength_from,
    )
    wf[NeXusDetectorName] = "detector"
    wf[NeXusName[FrameMonitor0]] = "monitor"
    wf[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': np.inf,
        'monitor': np.inf,
    }
    wf[EmptyDetector[SampleRun]] = detector_geometry
    wf[NeXusData[snx.NXdetector, SampleRun]] = detector_data
    wf[EmptyMonitor[SampleRun, FrameMonitor0]] = monitor_geometry
    wf[NeXusData[FrameMonitor0, SampleRun]] = monitor_data
    wf[Position[snx.NXsample, SampleRun]] = sc.vector([0, 0, 77], unit='m')
    wf[Position[snx.NXsource, SampleRun]] = fakes.source_position()
    wf[DiskChoppers[SampleRun]] = fakes.psc_choppers()

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
def test_GenericUnwrapWorkflow_computes_wavelength(
    wavelength_from, detector_or_monitor, simulation_results_psc_choppers
):
    wf = _make_workflow(wavelength_from=wavelength_from)

    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    if detector_or_monitor == "monitor":
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])
        assert 'wavelength' in wavs.coords
    else:
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
        assert 'wavelength' in wavs.bins.coords


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
def test_GenericUnwrapWorkflow_makes_different_luts_for_detector_and_monitor(
    wavelength_from, simulation_results_psc_choppers
):
    wf = _make_workflow(wavelength_from=wavelength_from)
    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    det_table = wf.compute(unwrap.LookupTable[SampleRun, snx.NXdetector])
    mon_table = wf.compute(unwrap.LookupTable[SampleRun, FrameMonitor0])

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
