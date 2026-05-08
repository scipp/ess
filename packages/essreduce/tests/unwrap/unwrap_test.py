# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scippneutron.chopper import DiskChopper

from ess.reduce import unwrap
from ess.reduce.nexus.types import (
    AnyRun,
    FrameMonitor0,
    NeXusDetectorName,
    NeXusName,
    RawDetector,
    RawMonitor,
    SampleRun,
)
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    LookupTableFromTof,
    LookupTableWorkflow,
    fakes,
)

sl = pytest.importorskip("sciline")


def make_lut_workflow(engine, choppers, pulse_stride, neutrons=None, seed=None):
    lut_wf = LookupTableFromTof() if engine == "tof" else LookupTableWorkflow()
    lut_wf[unwrap.DiskChoppers[AnyRun]] = choppers
    lut_wf[unwrap.SourcePosition] = fakes.source_position()
    lut_wf[unwrap.NumberOfSimulatedNeutrons] = neutrons
    lut_wf[unwrap.PulseStride] = pulse_stride
    if engine == "tof":
        lut_wf[unwrap.SimulationSeed] = seed
        lut_wf[unwrap.SimulationResults] = lut_wf.compute(unwrap.SimulationResults)
    return lut_wf


@pytest.fixture(scope="module")
def lut_workflow_psc_choppers():
    choppers = fakes.psc_choppers()
    return {
        'tof': make_lut_workflow(
            engine='tof', choppers=choppers, neutrons=1e6, seed=1234, pulse_stride=1
        ),
        'analytical': make_lut_workflow(
            engine='analytical', choppers=choppers, pulse_stride=1
        ),
    }


@pytest.fixture(scope="module")
def lut_workflow_pulse_skipping():
    choppers = fakes.pulse_skipping_choppers()
    return {
        'tof': make_lut_workflow(
            engine='tof', choppers=choppers, neutrons=1e6, seed=112, pulse_stride=2
        ),
        'analytical': make_lut_workflow(
            engine='analytical', choppers=choppers, pulse_stride=2
        ),
    }


def _make_workflow_event_mode(
    distance,
    choppers,
    lut_workflow,
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

    pl = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor0])
    if detector_or_monitor == "detector":
        pl[NeXusDetectorName] = "detector"
        pl[RawDetector[SampleRun]] = mon
        pl[unwrap.DetectorLtotal[SampleRun]] = distance
    else:
        pl[NeXusName[FrameMonitor0]] = "monitor"
        pl[RawMonitor[SampleRun, FrameMonitor0]] = mon
        pl[unwrap.MonitorLtotal[SampleRun, FrameMonitor0]] = distance

    pl[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': error_threshold,
        'monitor': error_threshold,
    }
    pl[unwrap.PulseStrideOffset] = pulse_stride_offset

    lut_wf = lut_workflow.copy()
    lut_wf[unwrap.LtotalRange] = distance, distance

    pl[unwrap.LookupTable] = lut_wf.compute(unwrap.LookupTable)

    return pl, ref


def _make_workflow_histogram_mode(
    dim, distance, choppers, lut_workflow, seed, error_threshold, detector_or_monitor
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

    pl = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor0])
    if detector_or_monitor == "detector":
        pl[NeXusDetectorName] = "detector"
        pl[RawDetector[SampleRun]] = mon
        pl[unwrap.DetectorLtotal[SampleRun]] = distance
    else:
        pl[NeXusName[FrameMonitor0]] = "monitor"
        pl[RawMonitor[SampleRun, FrameMonitor0]] = mon
        pl[unwrap.MonitorLtotal[SampleRun, FrameMonitor0]] = distance

    pl[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': error_threshold,
        'monitor': error_threshold,
    }

    lut_wf = lut_workflow.copy()
    lut_wf[unwrap.LtotalRange] = distance, distance

    pl[unwrap.LookupTable] = lut_wf.compute(unwrap.LookupTable)

    return pl, ref


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

    # graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    # wavs = tofs.transform_coords("wavelength", graph=graph)
    ref = ref.hist(wavelength=wavs.coords["wavelength"])
    # We divide by the maximum to avoid large relative differences at the edges of the
    # frames where the counts are low.
    diff = (wavs - ref) / ref.max()
    assert np.nanpercentile(diff.values, percentile) < diff_threshold
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN tof from the lookup).
    assert sc.isclose(ref.data.nansum(), wavs.data.nansum(), rtol=sc.scalar(rtol))


@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_unwrap_with_no_choppers(engine, detector_or_monitor) -> None:
    # At this small distance the frames are not overlapping (with the given wavelength
    # range), despite not using any choppers.
    distance = sc.scalar(10.0, unit="m")
    choppers = {}

    lut_wf = make_lut_workflow(
        engine=engine, choppers=choppers, neutrons=300_000, seed=1234, pulse_stride=1
    )

    pl, ref = _make_workflow_event_mode(
        distance=distance,
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=1,
        pulse_stride_offset=0,
        error_threshold=1.0,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs, ref=ref, percentile=96, diff_threshold=1.0, rtol=0.02
    )


# At 30m, event_time_offset does not wrap around (all events within the first pulse).
# At 60m, all events are within the second pulse.
# At 80m, events are split between the second and third pulse.
# At 108m, events are split between the third and fourth pulse.
# @pytest.mark.parametrize("dist", [30.0, 60.0, 80.0, 108.0])
@pytest.mark.parametrize("dist", [25.0, 50.0, 62.0, 90.0])
@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_standard_unwrap(
    dist, engine, detector_or_monitor, lut_workflow_psc_choppers
) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.psc_choppers(),
        lut_workflow=lut_workflow_psc_choppers[engine],
        seed=7,
        pulse_stride_offset=0,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.02,
        rtol=0.06 if engine == "tof" else 0.01,
    )


# At 30m, event_time_offset does not wrap around (all events within the first pulse).
# At 60m, all events are within the second pulse.
# At 80m, events are split between the second and third pulse.
# At 108m, events are split between the third and fourth pulse.
# @pytest.mark.parametrize("dist", [30.0, 60.0, 80.0, 108.0])
@pytest.mark.parametrize("dist", [25.0, 50.0, 62.0, 90.0])
@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("dim", ["time_of_flight", "tof", "frame_time"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_standard_unwrap_histogram_mode(
    dist, engine, dim, detector_or_monitor, lut_workflow_psc_choppers
) -> None:
    pl, ref = _make_workflow_histogram_mode(
        dim=dim,
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.psc_choppers(),
        lut_workflow=lut_workflow_psc_choppers[engine],
        seed=37,
        error_threshold=np.inf,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_histogram_mode(
        wavs=wavs,
        ref=ref,
        percentile=96,
        diff_threshold=0.4,
        rtol=0.06 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("dist", [60.0, 100.0])
@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap(
    dist, engine, detector_or_monitor, lut_workflow_pulse_skipping
) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        lut_workflow=lut_workflow_pulse_skipping[engine],
        seed=432,
        pulse_stride_offset=1,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("engine", ["tof", "analytical"])
def test_pulse_skipping_unwrap_180_phase_shift(engine, detector_or_monitor) -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers["pulse_skipping"].phase.value += 180.0

    lut_wf = make_lut_workflow(
        engine=engine, choppers=choppers, neutrons=500_000, seed=111, pulse_stride=2
    )

    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(100.0, unit="m"),
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=55,
        pulse_stride_offset=1,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("dist", [60.0, 100.0])
@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_stride_offset_guess_gives_expected_result(
    dist, engine, detector_or_monitor, lut_workflow_pulse_skipping
) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        lut_workflow=lut_workflow_pulse_skipping[engine],
        seed=97,
        pulse_stride_offset=None,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap_when_all_neutrons_arrive_after_second_pulse(
    engine, detector_or_monitor
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

    lut_wf = make_lut_workflow(
        engine=engine, choppers=choppers, neutrons=500_000, seed=222, pulse_stride=2
    )

    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(130.0, unit="m"),
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=6,
        pulse_stride_offset=1,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap_when_first_half_of_first_pulse_is_missing(
    engine, detector_or_monitor
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

    lut_wf = make_lut_workflow(
        engine=engine, choppers=choppers, neutrons=300_000, seed=1234, pulse_stride=2
    )
    lut_wf[unwrap.LtotalRange] = distance, distance

    pl = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor0])

    # Skip first pulse = half of the first frame
    a = mon.group('event_time_zero')['event_time_zero', 1:]
    a.bins.coords['event_time_zero'] = sc.bins_like(a, a.coords['event_time_zero'])
    concatenated = a.bins.concat('event_time_zero')

    pl[unwrap.LookupTable] = lut_wf.compute(unwrap.LookupTable)
    pl[unwrap.PulseStrideOffset] = 1  # Start the stride at the second pulse
    pl[unwrap.LookupTableRelativeErrorThreshold] = {
        'detector': np.inf,
        'monitor': np.inf,
    }

    if detector_or_monitor == "detector":
        pl[NeXusDetectorName] = "detector"
        pl[RawDetector[SampleRun]] = concatenated
        pl[unwrap.DetectorLtotal[SampleRun]] = distance
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        pl[NeXusName[FrameMonitor0]] = "monitor"
        pl[RawMonitor[SampleRun, FrameMonitor0]] = concatenated
        pl[unwrap.MonitorLtotal[SampleRun, FrameMonitor0]] = distance
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    # Convert to wavelength
    # graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
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
        pl.compute(target).data.nansum(),
        wavs.data.nansum(),
        rtol=sc.scalar(1.0e-3),
    )


@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_stride_3(engine, detector_or_monitor) -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers["pulse_skipping"].frequency.value = -14.0 / 3.0

    lut_wf = make_lut_workflow(
        engine=engine, choppers=choppers, neutrons=500_000, seed=111, pulse_stride=3
    )

    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(150.0, unit="m"),
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=68,
        pulse_stride_offset=None,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.1,
        rtol=0.05 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_pulse_skipping_unwrap_histogram_mode(
    engine, detector_or_monitor, lut_workflow_pulse_skipping
) -> None:
    pl, ref = _make_workflow_histogram_mode(
        dim='time_of_flight',
        distance=sc.scalar(50.0, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        lut_workflow=lut_workflow_pulse_skipping[engine],
        seed=9,
        error_threshold=np.inf,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_histogram_mode(
        wavs=wavs,
        ref=ref,
        percentile=96,
        diff_threshold=0.4,
        rtol=0.05 if engine == "tof" else 0.01,
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("engine", ["tof", "analytical"])
@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
def test_unwrap_int(
    dtype, engine, detector_or_monitor, lut_workflow_psc_choppers
) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(62.0, unit="m"),
        choppers=fakes.psc_choppers(),
        lut_workflow=lut_workflow_psc_choppers[engine],
        seed=2,
        pulse_stride_offset=0,
        error_threshold=0.1,
        detector_or_monitor=detector_or_monitor,
    )

    if detector_or_monitor == "detector":
        target = RawDetector[SampleRun]
    else:
        target = RawMonitor[SampleRun, FrameMonitor0]
    mon = pl.compute(target).copy()
    mon.bins.coords["event_time_offset"] = mon.bins.coords["event_time_offset"].to(
        dtype=dtype, unit="ns"
    )
    pl[target] = mon

    if detector_or_monitor == "detector":
        wavs = pl.compute(unwrap.WavelengthDetector[SampleRun])
    else:
        wavs = pl.compute(unwrap.WavelengthMonitor[SampleRun, FrameMonitor0])

    _validate_result_events(
        wavs=wavs,
        ref=ref,
        percentile=100,
        diff_threshold=0.02,
        rtol=0.05 if engine == "tof" else 0.01,
    )
