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
    AnyRun,
    DiskChoppers,
    EmptyDetector,
    FrameMonitor0,
    NeXusData,
    NeXusDetectorName,
    NeXusName,
    Position,
    RawDetector,
    SampleRun,
)
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    LookupTableWorkflow,
    fakes,
    load_lookup_table_from_file,
)
from ess.reduce.unwrap.lut import lut_from_simulation_providers

sl = pytest.importorskip("sciline")


@pytest.fixture
def workflow() -> sciline.Pipeline:
    sizes = {'detector_number': 10}
    calibrated_beamline = sc.DataArray(
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

    events = sc.DataArray(
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
    nexus_data = sc.DataArray(
        sc.bins(
            begin=sc.array(dims=["pulse"], values=[0], unit=None),
            data=events,
            dim="event",
        )
    )

    wf = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor0])
    wf[NeXusDetectorName] = "detector"
    wf[NeXusName[FrameMonitor0]] = "monitor"
    wf[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': np.inf,
        'monitor': np.inf,
    }
    wf[EmptyDetector[SampleRun]] = calibrated_beamline
    wf[NeXusData[snx.NXdetector, SampleRun]] = nexus_data
    wf[Position[snx.NXsample, SampleRun]] = sc.vector([0, 0, 77], unit='m')
    wf[Position[snx.NXsource, SampleRun]] = sc.vector([0, 0, 0], unit='m')

    return wf


# def test_LookupTableWorkflow_can_compute_lut():
#     wf = LookupTableWorkflow()
#     wf[DiskChoppers[AnyRun]] = fakes.psc_choppers()
#     wf[unwrap.NumberOfSimulatedNeutrons] = 10_000
#     wf[unwrap.LtotalRange] = (
#         sc.scalar(75.0, unit="m"),
#         sc.scalar(85.0, unit="m"),
#     )
#     wf[unwrap.SourcePosition] = fakes.source_position()
#     lut = wf.compute(unwrap.LookupTable)
#     assert lut.array is not None
#     assert lut.distance_resolution is not None
#     assert lut.time_resolution is not None
#     assert lut.pulse_stride is not None
#     assert lut.pulse_period is not None
#     assert lut.choppers is not None


def test_GenericUnwrapWorkflow_with_lut_from_tof_simulation(workflow):
    for provider in lut_from_simulation_providers():
        workflow.insert(provider)

    # Should be able to compute DetectorData without chopper and simulation params
    # This contains event_time_offset (time-of-arrival).
    _ = workflow.compute(RawDetector[SampleRun])

    workflow[DiskChoppers[SampleRun]] = fakes.psc_choppers()
    workflow[unwrap.NumberOfSimulatedNeutrons] = 10_000
    workflow[unwrap.SimulationSeed] = None
    workflow[unwrap.SimulationFacility] = 'ess'

    detector = workflow.compute(unwrap.WavelengthDetector[SampleRun])
    assert 'wavelength' in detector.bins.coords


def test_GenericUnwrapWorkflow_with_lut_from_polygons(workflow):
    # Should be able to compute DetectorData without chopper params
    _ = workflow.compute(RawDetector[SampleRun])

    workflow[DiskChoppers[SampleRun]] = fakes.psc_choppers()

    detector = workflow.compute(unwrap.WavelengthDetector[SampleRun])
    assert 'wavelength' in detector.bins.coords


@pytest.mark.parametrize("engine", ["tof", "analytical"])
def test_GenericUnwrapWorkflow_with_lut_from_file(
    engine, workflow, tmp_path: pytest.TempPathFactory
):
    lut_wf = LookupTableWorkflow(use_simulation=(engine == "tof"))
    lut_wf[DiskChoppers[AnyRun]] = fakes.psc_choppers()
    if engine == "tof":
        lut_wf[unwrap.NumberOfSimulatedNeutrons] = 10_000
    lut_wf[unwrap.LtotalRange[AnyRun, snx.NXdetector]] = (
        sc.scalar(75.0, unit="m"),
        sc.scalar(85.0, unit="m"),
    )
    lut_wf[Position[snx.NXsource, AnyRun]] = fakes.source_position()
    lut = lut_wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])
    lut.save_hdf5(filename=tmp_path / "lut.h5")

    workflow.insert(load_lookup_table_from_file)
    workflow[unwrap.LookupTableFilename[AnyRun, snx.NXdetector]] = (
        tmp_path / "lut.h5"
    ).as_posix()

    loaded_lut = workflow.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])
    assert_identical(lut.array, loaded_lut.array)
    assert_identical(lut.pulse_period, loaded_lut.pulse_period)
    assert lut.pulse_stride == loaded_lut.pulse_stride
    assert_identical(lut.distance_resolution, loaded_lut.distance_resolution)
    assert_identical(lut.time_resolution, loaded_lut.time_resolution)
    assert_identical(lut.choppers, loaded_lut.choppers)

    detector = workflow.compute(unwrap.WavelengthDetector[SampleRun])
    assert 'wavelength' in detector.bins.coords


def test_GenericUnwrapWorkflow_with_lut_from_file_old_format(
    workflow, tmp_path: pytest.TempPathFactory
):
    lut_wf = LookupTableWorkflow(use_simulation=False)
    lut_wf[DiskChoppers[AnyRun]] = fakes.psc_choppers()
    lut_wf[unwrap.NumberOfSimulatedNeutrons] = 10_000
    lut_wf[unwrap.LtotalRange] = (
        sc.scalar(75.0, unit="m"),
        sc.scalar(85.0, unit="m"),
    )
    lut_wf[Position[snx.NXsource, AnyRun]] = fakes.source_position()
    lut = lut_wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])
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

    workflow[unwrap.LookupTableFilename[AnyRun, snx.NXdetector]] = (
        tmp_path / "lut.h5"
    ).as_posix()
    loaded_lut = workflow.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])
    assert_identical(lut.array, loaded_lut.array)
    assert_identical(lut.pulse_period, loaded_lut.pulse_period)
    assert lut.pulse_stride == loaded_lut.pulse_stride
    assert_identical(lut.distance_resolution, loaded_lut.distance_resolution)
    assert_identical(lut.time_resolution, loaded_lut.time_resolution)
    assert loaded_lut.choppers is None  # No chopper info in old format

    detector = workflow.compute(unwrap.WavelengthDetector[SampleRun])
    assert 'wavelength' in detector.bins.coords
