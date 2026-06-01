# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scippneutron.chopper import DiskChopper
from scippnexus import NXsource

from ess.reduce import unwrap
from ess.reduce.nexus.types import NeXusDetectorName, Position, RawDetector, SampleRun
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    fakes,
    simulate_chopper_cascade_using_tof,
)

sl = pytest.importorskip("sciline")


def dream_choppers() -> dict[str, DiskChopper]:
    psc1 = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(286 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -70.405], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=[-1.23, 70.49, 84.765, 113.565, 170.29, 271.635, 286.035, 301.17],
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=[1.23, 73.51, 88.035, 116.835, 175.31, 275.565, 289.965, 303.63],
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    psc2 = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-236, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -70.395], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=[-1.23, 27.0, 55.8, 142.385, 156.765, 214.115, 257.23, 315.49],
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=[1.23, 30.6, 59.4, 145.615, 160.035, 217.885, 261.17, 318.11],
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    oc = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(297 - 180 - 90, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -70.376], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-27.6 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[27.6 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    bcc = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -66.77], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[36.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    t0 = DiskChopper(
        frequency=sc.scalar(28.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(280 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -63.5], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-314.9 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[314.9 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    return {"psc1": psc1, "psc2": psc2, "oc": oc, "bcc": bcc, "t0": t0}


def dream_choppers_with_frame_overlap() -> dict[str, DiskChopper]:
    out = dream_choppers()
    out["bcc"] = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -66.77], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[56.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )
    return out


def dream_source_position() -> sc.Variable:
    return sc.vector(value=[0, 0, -76.55], unit="m")


def simulate_with_tof(choppers, pulse_stride, source_position):
    return simulate_chopper_cascade_using_tof(
        choppers=choppers,
        source_position=source_position,
        neutrons=300_000,
        pulse_stride=pulse_stride,
        seed=432,
        facility="ess",
    )


@pytest.fixture(scope="module")
def simulate_with_dream_choppers() -> dict[str, sl.Pipeline]:
    return simulate_with_tof(
        choppers=dream_choppers(),
        pulse_stride=1,
        source_position=dream_source_position(),
    )


def setup_workflow(
    wavelength_from: str,
    raw_data: sc.DataArray,
    ltotal: sc.Variable,
    choppers: dict[str, DiskChopper],
    source_position: sc.Variable,
    error_threshold: float = 0.1,
) -> sl.Pipeline:
    wf = GenericUnwrapWorkflow(run_types=[SampleRun], monitor_types=[], wavelength_from=wavelength_from)
    wf[RawDetector[SampleRun]] = raw_data
    wf[unwrap.DetectorLtotal[SampleRun]] = ltotal
    wf[NeXusDetectorName] = "detector"
    wf[unwrap.LookupTableRelativeErrorThreshold] = {"detector": error_threshold}
    wf[unwrap.DiskChoppers[SampleRun]] = choppers
    wf[Position[NXsource, SampleRun]] = source_position
    return wf


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=["detector_number"], values=[77.675], unit="m"),
        sc.array(dims=["detector_number"], values=[77.675, 76.5], unit="m"),
        sc.array(
            dims=["y", "x"],
            values=[[77.675, 76.1, 78.05], [77.15, 77.3, 77.675]],
            unit="m",
        ),
    ],
)
@pytest.mark.parametrize("time_offset_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("distance_unit", ["m", "mm"])
def test_dream_wfm(
    wavelength_from, ltotal, time_offset_unit, distance_unit, simulate_with_dream_choppers
):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    choppers = dream_choppers()
    source_position = dream_source_position()

    # Create some neutron events
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=10_000,
        seed=77,
        source_position=source_position,
    )

    raw = sc.concat(
        [beamline.get_monitor(key)[0].squeeze() for key in monitors.keys()],
        dim="detector",
    ).fold(dim="detector", sizes=ltotal.sizes)

    # Convert the time offset to the unit requested by the test
    raw.bins.coords["event_time_offset"] = raw.bins.coords["event_time_offset"].to(
        unit=time_offset_unit, copy=False
    )
    # Convert the distance to the unit requested by the test
    raw.coords["Ltotal"] = raw.coords["Ltotal"].to(unit=distance_unit, copy=False)

    # Save reference data
    ref = beamline.get_monitor(next(iter(monitors)))[1].squeeze()
    ref = sc.sort(ref, key='id')

    wf = setup_workflow(
        wavelength_from=wavelength_from,
        raw_data=raw,
        ltotal=ltotal,
        choppers=choppers,
        source_position=source_position,
    )

    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_dream_choppers

    wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        diff = abs(
            (x.coords["wavelength"] - ref.coords["wavelength"])
            / ref.coords["wavelength"]
        )
        assert np.nanpercentile(diff.values, 99.9) < 0.02
        assert sc.isclose(ref.data.sum(), da.data.sum(), rtol=sc.scalar(1.0e-3))


@pytest.fixture(scope="module")
def simulate_with_dream_choppers_time_overlap() -> dict[str, sl.Pipeline]:
    return simulate_with_tof(
        choppers=dream_choppers_with_frame_overlap(),
        pulse_stride=1,
        source_position=dream_source_position(),
    )


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=["detector_number"], values=[77.675], unit="m"),
        sc.array(dims=["detector_number"], values=[77.675, 76.5], unit="m"),
        sc.array(
            dims=["y", "x"],
            values=[[77.675, 76.1, 78.05], [77.15, 77.3, 77.675]],
            unit="m",
        ),
    ],
)
@pytest.mark.parametrize("time_offset_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("distance_unit", ["m", "mm"])
def test_dream_wfm_with_subframe_time_overlap(
    wavelength_from,
    ltotal,
    time_offset_unit,
    distance_unit,
    simulate_with_dream_choppers_time_overlap,
):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    choppers = dream_choppers_with_frame_overlap()
    source_position = dream_source_position()

    # Create some neutron events
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=10_000,
        seed=88,
        source_position=source_position,
    )

    raw = sc.concat(
        [beamline.get_monitor(key)[0].squeeze() for key in monitors.keys()],
        dim="detector",
    ).fold(dim="detector", sizes=ltotal.sizes)

    # Convert the time offset to the unit requested by the test
    raw.bins.coords["event_time_offset"] = raw.bins.coords["event_time_offset"].to(
        unit=time_offset_unit, copy=False
    )
    # Convert the distance to the unit requested by the test
    raw.coords["Ltotal"] = raw.coords["Ltotal"].to(unit=distance_unit, copy=False)

    # Save reference data
    ref = beamline.get_monitor(next(iter(monitors)))[1].squeeze()
    ref = sc.sort(ref, key='id')

    wf = setup_workflow(
        wavelength_from=wavelength_from,
        raw_data=raw,
        ltotal=ltotal,
        choppers=choppers,
        source_position=source_position,
        error_threshold=0.01,
    )
    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = (
            simulate_with_dream_choppers_time_overlap
        )

    wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        sel = sc.isfinite(x.coords["wavelength"])
        y = ref.coords["wavelength"][sel]
        diff = abs((x.coords["wavelength"][sel] - y) / y)
        assert np.nanpercentile(diff.values, 99.9) < 0.02
        sum_wfm = da.hist(wavelength=100).data.sum()
        sum_ref = ref.hist(wavelength=100).data.sum()
        # Verify that we lost some neutrons that were in the overlapping region
        assert sum_wfm < sum_ref
        assert sum_wfm > sum_ref * 0.8


def v20_choppers():
    wfm1 = DiskChopper(
        frequency=sc.scalar(-70.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-47.10, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.6], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([83.71, 140.49, 193.26, 242.32, 287.91, 330.3]) + 15.0,
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([94.7, 155.79, 212.56, 265.33, 314.37, 360.0]) + 15.0,
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    wfm2 = DiskChopper(
        frequency=sc.scalar(-70.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-76.76, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 7.1], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([65.04, 126.1, 182.88, 235.67, 284.73, 330.32]) + 15.0,
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([76.03, 141.4, 202.18, 254.97, 307.74, 360.0]) + 15.0,
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    foc1 = DiskChopper(
        frequency=sc.scalar(-56.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-62.40, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 8.8], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([74.6, 139.6, 194.3, 245.3, 294.8, 347.2]),
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([95.2, 162.8, 216.1, 263.1, 310.5, 371.6]),
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    foc2 = DiskChopper(
        frequency=sc.scalar(-28.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-12.27, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 15.9], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([98.0, 154.0, 206.8, 255.0, 299.0, 344.65]),
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([134.6, 190.06, 237.01, 280.88, 323.56, 373.76]),
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    pol = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(0.0, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 17.0], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([40.0]),
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([240.0]),
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )
    return {"wfm1": wfm1, "wfm2": wfm2, "foc1": foc1, "foc2": foc2, "pol": pol}


def v20_source_position():
    return sc.vector([0, 0, 0], unit='m')


@pytest.fixture(scope="module")
def simulate_with_v20_choppers() -> dict[str, sl.Pipeline]:
    return simulate_with_tof(
        choppers=v20_choppers(),
        pulse_stride=1,
        source_position=v20_source_position(),
    )


@pytest.mark.parametrize("wavelength_from", ["simulation", "analytical"])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=["detector_number"], values=[26.0], unit="m"),
        sc.array(dims=["detector_number"], values=[26.0, 25.5], unit="m"),
        sc.array(
            dims=["y", "x"], values=[[26.0, 25.1, 26.33], [25.9, 26.0, 25.7]], unit="m"
        ),
    ],
)
@pytest.mark.parametrize("time_offset_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("distance_unit", ["m", "mm"])
def test_v20_compute_wavelengths_from_wfm(
    wavelength_from, ltotal, time_offset_unit, distance_unit, simulate_with_v20_choppers
):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    choppers = v20_choppers()
    source_position = v20_source_position()

    # Create some neutron events
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=10_000,
        seed=99,
        source_position=source_position,
    )

    raw = sc.concat(
        [beamline.get_monitor(key)[0].squeeze() for key in monitors.keys()],
        dim="detector",
    ).fold(dim="detector", sizes=ltotal.sizes)

    # Convert the time offset to the unit requested by the test
    raw.bins.coords["event_time_offset"] = raw.bins.coords["event_time_offset"].to(
        unit=time_offset_unit, copy=False
    )
    # Convert the distance to the unit requested by the test
    raw.coords["Ltotal"] = raw.coords["Ltotal"].to(unit=distance_unit, copy=False)

    # Save reference data
    ref = beamline.get_monitor(next(iter(monitors)))[1].squeeze()
    ref = sc.sort(ref, key='id')

    wf = setup_workflow(
        wavelength_from=wavelength_from,
        raw_data=raw,
        ltotal=ltotal,
        choppers=choppers,
        source_position=source_position,
    )
    if wavelength_from == "simulation":
        wf[unwrap.SimulationResults[SampleRun]] = simulate_with_v20_choppers

    wavs = wf.compute(unwrap.WavelengthDetector[SampleRun])

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        diff = abs(
            (x.coords["wavelength"] - ref.coords["wavelength"])
            / ref.coords["wavelength"]
        )
        if wavelength_from == "simulation":
            assert np.nanpercentile(diff.values, 99) < 0.02
        else:
            assert np.nanpercentile(diff.values, 90) < 0.05
        assert sc.isclose(ref.data.sum(), da.data.sum(), rtol=sc.scalar(1.0e-3))
