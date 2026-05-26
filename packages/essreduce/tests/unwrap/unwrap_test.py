# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scippneutron.chopper import DiskChopper
from scippnexus import NXsource

from ess.reduce import unwrap
from ess.reduce.nexus.types import (
    FrameMonitor0,
    NeXusDetectorName,
    NeXusName,
    Position,
    RawDetector,
    RawMonitor,
    SampleRun,
)
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    fakes,
    simulate_chopper_cascade_using_tof,
)

sl = pytest.importorskip("sciline")


def simulate_with_tof(choppers, pulse_stride, neutrons=None, seed=None):
    return simulate_chopper_cascade_using_tof(
        choppers=choppers,
        source_position=fakes.source_position(),
        neutrons=neutrons,
        pulse_stride=pulse_stride,
        seed=seed,
        facility="ess",
    )


@pytest.fixture(scope="module")
def simulation_results_psc_choppers():
    return simulate_with_tof(
        choppers=fakes.psc_choppers(), pulse_stride=1, neutrons=1e6, seed=1234
    )


@pytest.fixture(scope="module")
def simulation_results_pulse_skipping():
    return simulate_with_tof(
        choppers=fakes.pulse_skipping_choppers(),
        pulse_stride=2,
        neutrons=1e6,
        seed=112,
    )


def _initialize_workflow(mode, distance, error_threshold, choppers):
    wf = GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[FrameMonitor0], mode=mode
    )
    wf[NeXusDetectorName] = "detector"
    wf[unwrap.DetectorLtotal[SampleRun]] = distance
    wf[NeXusName[FrameMonitor0]] = "monitor"
    wf[unwrap.MonitorLtotal[SampleRun, FrameMonitor0]] = distance

    wf[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': error_threshold,
        'monitor': error_threshold,
    }
    wf[unwrap.DiskChoppers[SampleRun]] = choppers
    wf[Position[NXsource, SampleRun]] = fakes.source_position()
    return wf


def _make_workflow_event_mode(
    mode,
    distance,
    choppers,
    seed,
    pulse_stride_offset,
    error_threshold,
    detector_or_monitor,
):
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=300_000,
        seed=seed,
    )
    mon, ref = beamline.get_monitor("detector")

    wf = _initialize_workflow(
        mode=mode, distance=distance, error_threshold=error_threshold, choppers=choppers
    )

    if detector_or_monitor == "detector":
        wf[RawDetector[SampleRun]] = mon
    else:
        wf[RawMonitor[SampleRun, FrameMonitor0]] = mon

    wf[unwrap.PulseStrideOffset] = pulse_stride_offset

    return wf, ref


def _make_workflow_histogram_mode(
    mode, dim, distance, choppers, seed, error_threshold, detector_or_monitor
):
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
        seed=seed,
    )
    mon, ref = beamline.get_monitor("detector")
    mon = mon.hist(
        event_time_offset=sc.linspace(
            "event_time_offset", 0.0, 1000.0 / 14, num=301, unit="ms"
        ).to(unit=mon.bins.coords["event_time_offset"].bins.unit)
    ).rename(event_time_offset=dim)

    wf = _initialize_workflow(
        mode=mode, distance=distance, error_threshold=error_threshold, choppers=choppers
    )

    if detector_or_monitor == "detector":
        wf[RawDetector[SampleRun]] = mon
    else:
        wf[RawMonitor[SampleRun, FrameMonitor0]] = mon

    return wf, ref


def _validate_result_events(wavs, ref, percentile, diff_threshold, rtol):
    assert "event_time_offset" not in wavs.coords
    assert "tof" not in wavs.coords

    wavs = wavs.bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # Most errors should be small
    assert np.nanpercentile(diff.values, percentile) < diff_threshold
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN tof from the lookup).
    nevents = sc.sum(~sc.isnan(wavs.coords['wavelength'])).to(dtype=float)
    nevents.unit = 'counts'
    assert sc.isclose(ref.data.sum(), nevents, rtol=sc.scalar(rtol))


def _validate_result_histogram_mode(wavs, ref, percentile, diff_threshold, rtol):
    assert "tof" not in wavs.coords
    assert "time_of_flight" not in wavs.coords
    assert "frame_time" not in wavs.coords

    ref = ref.hist(wavelength=wavs.coords["wavelength"])
    # We divide by the maximum to avoid large relative differences at the edges of the
    # frames where the counts are low.
    diff = (wavs - ref) / ref.max()
    assert np.nanpercentile(diff.values, percentile) < diff_threshold
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN tof from the lookup).
    assert sc.isclose(ref.data.nansum(), wavs.data.nansum(), rtol=sc.scalar(rtol))


@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_unwrap_with_no_choppers(mode, detector_or_monitor) -> None:
    # At this small distance the frames are not overlapping (with the given wavelength
    # range), despite not using any choppers.
    distance = sc.scalar(10.0, unit="m")
    choppers = {}

    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=distance,
        choppers=choppers,
        seed=1,
        pulse_stride_offset=0,
        error_threshold=1.0,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_tof(
            choppers=choppers, pulse_stride=1, neutrons=300_000, seed=1234
        )

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs, ref=ref, percentile=96, diff_threshold=1.0, rtol=0.02
    )


# At 25m, event_time_offset does not wrap around (all events within the first pulse).
# At 50m, all events are within the second pulse.
# At 62m, events are split between the second and third pulse.
# At 90m, events are split between the third and fourth pulse.
@pytest.mark.parametrize("dist", [25.0, 50.0, 62.0, 90.0])
@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_standard_unwrap(
    dist, mode, detector_or_monitor, simulation_results_psc_choppers
) -> None:
    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.psc_choppers(),
        seed=7,
        pulse_stride_offset=0,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.02,
        rtol=0.06 if mode == "simulation" else 0.01,
    )


# At 25m, event_time_offset does not wrap around (all events within the first pulse).
# At 50m, all events are within the second pulse.
# At 62m, events are split between the second and third pulse.
# At 90m, events are split between the third and fourth pulse.
@pytest.mark.parametrize("dist", [25.0, 50.0, 62.0, 90.0])
@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("dim", ["time_of_flight", "tof", "frame_time"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_standard_unwrap_histogram_mode(
    dist, mode, dim, detector_or_monitor, simulation_results_psc_choppers
) -> None:
    wf, ref = _make_workflow_histogram_mode(
        mode=mode,
        dim=dim,
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.psc_choppers(),
        seed=37,
        error_threshold=np.inf,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_histogram_mode(
        wavs=wavs,
        ref=ref,
        percentile=96,
        diff_threshold=0.4,
        rtol=0.06 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("dist", [60.0, 100.0])
@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap(
    dist, mode, detector_or_monitor, simulation_results_pulse_skipping
) -> None:
    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        seed=432,
        pulse_stride_offset=1,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_pulse_skipping

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("mode", ["simulation", "analytical"])
def test_pulse_skipping_unwrap_180_phase_shift(mode, detector_or_monitor) -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers["pulse_skipping"].phase.value += 180.0

    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(100.0, unit="m"),
        choppers=choppers,
        seed=55,
        pulse_stride_offset=1,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_tof(
            choppers=choppers, pulse_stride=2, neutrons=500_000, seed=111
        )

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("dist", [60.0, 100.0])
@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_stride_offset_guess_gives_expected_result(
    dist, mode, detector_or_monitor, simulation_results_pulse_skipping
) -> None:
    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        seed=97,
        pulse_stride_offset=None,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_pulse_skipping

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap_when_all_neutrons_arrive_after_second_pulse(
    mode, detector_or_monitor
) -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers['chopper'] = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-35.0, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 8.0], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=np.array([10.0]), unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=np.array([25.0]), unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(130.0, unit="m"),
        choppers=choppers,
        seed=6,
        pulse_stride_offset=1,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_tof(
            choppers=choppers, pulse_stride=2, neutrons=500_000, seed=222
        )

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap_when_first_half_of_first_pulse_is_missing(
    mode, detector_or_monitor
) -> None:
    distance = sc.scalar(100.0, unit="m")
    choppers = fakes.pulse_skipping_choppers()

    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1.0, unit="s"),
        events_per_pulse=100_000,
        seed=21,
    )
    mon, ref = beamline.get_monitor("detector")

    wf = _initialize_workflow(
        mode=mode, distance=distance, error_threshold=np.inf, choppers=choppers
    )
    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_tof(
            choppers=choppers, pulse_stride=2, neutrons=300_000, seed=1234
        )

    # Skip first pulse = half of the first frame
    a = mon.group('event_time_zero')['event_time_zero', 1:]
    a.bins.coords['event_time_zero'] = sc.bins_like(a, a.coords['event_time_zero'])
    concatenated = a.bins.concat('event_time_zero')

    if detector_or_monitor == "detector":
        wf[RawDetector[SampleRun]] = concatenated
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wf[RawMonitor[SampleRun, FrameMonitor0]] = concatenated
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    wavs = wavs.bins.concat().value
    # Bin the events in toa starting from the pulse period to skip the first pulse.
    ref = (
        ref.bin(
            toa=sc.concat(
                [
                    sc.scalar(1 / 14, unit='s').to(unit=ref.coords['toa'].unit),
                    ref.coords['toa'].max() * 1.01,
                ],
                dim='toa',
            )
        )
        .bins.concat()
        .value
    )

    # Sort the events according id to make sure we are comparing the same values.
    wavs = sc.sort(wavs, key=wavs.coords['id'])
    ref = sc.sort(ref, key=ref.coords['id'])

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.06
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN wavelength from the lookup).
    if detector_or_monitor == "detector":
        target = RawDetector[SampleRun]
    else:
        target = RawMonitor[SampleRun, FrameMonitor0]
    assert sc.isclose(
        wf.compute(target).data.nansum(),
        wavs.data.nansum(),
        rtol=sc.scalar(1.0e-3),
    )


@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_stride_3(mode, detector_or_monitor) -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers["pulse_skipping"].frequency.value = -14.0 / 3.0

    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(150.0, unit="m"),
        choppers=choppers,
        seed=68,
        pulse_stride_offset=None,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_tof(
            choppers=choppers, pulse_stride=3, neutrons=500_000, seed=111
        )

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap_histogram_mode(
    mode, detector_or_monitor, simulation_results_pulse_skipping
) -> None:
    wf, ref = _make_workflow_histogram_mode(
        mode=mode,
        dim='time_of_flight',
        distance=sc.scalar(50.0, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        seed=9,
        error_threshold=np.inf,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_pulse_skipping

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_histogram_mode(
        wavs=wavs,
        ref=ref,
        percentile=96,
        diff_threshold=0.4,
        rtol=0.05 if mode == "simulation" else 0.01,
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("mode", ["simulation", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_unwrap_int(
    dtype, mode, detector_or_monitor, simulation_results_psc_choppers
) -> None:
    wf, ref = _make_workflow_event_mode(
        mode=mode,
        distance=sc.scalar(62.0, unit="m"),
        choppers=fakes.psc_choppers(),
        seed=2,
        pulse_stride_offset=0,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if mode == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulation_results_psc_choppers

    if detector_or_monitor == "detector":
        target = RawDetector[SampleRun]
    else:
        target = RawMonitor[SampleRun, FrameMonitor0]
    mon = wf.compute(target).copy()
    mon.bins.coords["event_time_offset"] = mon.bins.coords["event_time_offset"].to(
        dtype=dtype, unit="ns"
    )
    wf[target] = mon

    if detector_or_monitor == "detector":
        wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = wf.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.02,
        rtol=0.05 if mode == "simulation" else 0.01,
    )
