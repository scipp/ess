# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scippnexus as snx
from scippneutron.chopper import DiskChopper

from ess.reduce import unwrap
from ess.reduce.nexus.types import AnyRun, FrameMonitor0, Position
from ess.reduce.unwrap import GenericUnwrapWorkflow, LookupTableWorkflow, SourceBounds

sl = pytest.importorskip("sciline")


def _make_workflow(wavelength_from: str = "analytical") -> sl.Pipeline:
    return GenericUnwrapWorkflow(
        run_types=[AnyRun],
        monitor_types=[FrameMonitor0],
        wavelength_from=wavelength_from,
    )


@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_computes_table(detector_or_monitor, wavelength_from):
    wf = _make_workflow(wavelength_from)
    wf[unwrap.DiskChoppers[AnyRun]] = {}
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 0], unit='m')
    wf[unwrap.PulseStride[AnyRun]] = 1

    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 60

    lmin, lmax = sc.scalar(25.0, unit='m'), sc.scalar(35.0, unit='m')
    dres = sc.scalar(0.1, unit='m')
    tres = sc.scalar(333.0, unit='us')

    Comp = snx.NXdetector if detector_or_monitor == "detector" else FrameMonitor0

    wf[unwrap.LtotalRange[AnyRun, Comp]] = lmin, lmax
    wf[unwrap.DistanceResolution] = dres
    wf[unwrap.TimeResolution] = tres

    table = wf.compute(unwrap.LookupTable[AnyRun, Comp])

    assert table.array.coords['distance'].min() < lmin
    assert table.array.coords['distance'].max() > lmax
    assert table.array.coords['event_time_offset'].max() == sc.scalar(
        1 / 14, unit='s'
    ).to(unit=table.array.coords['event_time_offset'].unit)
    assert sc.isclose(table.distance_resolution, dres)
    # Note that the time resolution is not exactly preserved since we want the table to
    # span exactly the frame period.
    assert sc.isclose(table.time_resolution, tres, rtol=sc.scalar(0.01))


@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_pulse_skipping(detector_or_monitor, wavelength_from):
    wf = _make_workflow(wavelength_from)
    wf[unwrap.DiskChoppers[AnyRun]] = {}
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 0], unit='m')
    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 62
    wf[unwrap.PulseStride[AnyRun]] = 2

    lmin, lmax = sc.scalar(55.0, unit='m'), sc.scalar(65.0, unit='m')
    dres = sc.scalar(0.1, unit='m')
    tres = sc.scalar(250.0, unit='us')

    Comp = snx.NXdetector if detector_or_monitor == "detector" else FrameMonitor0

    wf[unwrap.LtotalRange[AnyRun, Comp]] = lmin, lmax
    wf[unwrap.DistanceResolution] = dres
    wf[unwrap.TimeResolution] = tres

    table = wf.compute(unwrap.LookupTable[AnyRun, Comp])

    assert table.array.coords['event_time_offset'].max() == 2 * sc.scalar(
        1 / 14, unit='s'
    ).to(unit=table.array.coords['event_time_offset'].unit)


@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_non_exact_distance_range(wavelength_from):
    wf = _make_workflow(wavelength_from)
    wf[unwrap.DiskChoppers[AnyRun]] = {}
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 0], unit='m')
    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 63
    wf[unwrap.PulseStride[AnyRun]] = 1

    lmin, lmax = sc.scalar(25.0, unit='m'), sc.scalar(35.0, unit='m')
    dres = sc.scalar(0.33, unit='m')
    tres = sc.scalar(250.0, unit='us')

    wf[unwrap.LtotalRange[AnyRun, snx.NXdetector]] = lmin, lmax
    wf[unwrap.DistanceResolution] = dres
    wf[unwrap.TimeResolution] = tres

    table = wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])

    assert table.array.coords['distance'].min() < lmin
    assert table.array.coords['distance'].max() > lmax
    assert sc.isclose(table.distance_resolution, dres)


def _make_choppers():
    return {
        'wfm1': DiskChopper(
            axle_position=sc.vector([0, 0, 6.85], unit='m'),
            frequency=sc.scalar(-56, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-93.244, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[-1.9419, 49.5756], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[1.9419, 55.7157], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'WFMC_2': DiskChopper(
            axle_position=sc.vector([0, 0, 7.15], unit='m'),
            frequency=sc.scalar(-56, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-97.128, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[-1.9419, 51.8318], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[1.9419, 57.9719], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'FOC_1': DiskChopper(
            axle_position=sc.vector([0, 0, 8.4], unit='m'),
            frequency=sc.scalar(-42, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-81.3033, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[-5.1362, 42.5536], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[5.1362, 54.2095], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'FOC_2': DiskChopper(
            axle_position=sc.vector([0, 0, 12.2], unit='m'),
            frequency=sc.scalar(-42, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-107.013, unit='deg'),
            slit_begin=sc.array(
                dims=["cutout"], values=[-16.3227, 53.7401], unit='deg'
            ),
            slit_end=sc.array(dims=["cutout"], values=[16.3227, 86.8303], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'FOC_5': DiskChopper(
            axle_position=sc.vector([0, 0, 33], unit='m'),
            frequency=sc.scalar(-14, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-82.581, unit='deg'),
            slit_begin=sc.array(
                dims=["cutout"], values=[-25.8514, 38.3239], unit='deg'
            ),
            slit_end=sc.array(dims=["cutout"], values=[25.8514, 88.4621], unit='deg'),
            slit_height=None,
            radius=None,
        ),
    }


@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_computes_table_with_choppers(
    detector_or_monitor, wavelength_from
):
    wf = _make_workflow(wavelength_from)
    wf[unwrap.DiskChoppers[AnyRun]] = _make_choppers()
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 0], unit='m')
    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 64
    wf[unwrap.PulseStride[AnyRun]] = 1

    Comp = snx.NXdetector if detector_or_monitor == "detector" else FrameMonitor0

    wf[unwrap.LtotalRange[AnyRun, Comp]] = (
        sc.scalar(35.0, unit='m'),
        sc.scalar(65.0, unit='m'),
    )
    wf[unwrap.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[unwrap.TimeResolution] = sc.scalar(250.0, unit='us')

    table = wf.compute(unwrap.LookupTable[AnyRun, Comp])

    # At low distance, the rays are more focussed
    low_dist = table.array['distance', 2]
    eto = low_dist.coords['event_time_offset'][sc.isfinite(low_dist.data)]
    assert eto.min() > sc.scalar(1.0e4, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(1.5e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(3.3e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(3.8e4, unit="us").to(unit=eto.unit)

    # At high distance, the rays are more spread out
    high_dist = table.array['distance', -3]
    eto = high_dist.coords['event_time_offset'][sc.isfinite(high_dist.data)]
    assert eto.min() > sc.scalar(1.7e4, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(2.2e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(6.4e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(6.9e4, unit="us").to(unit=eto.unit)


@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_computes_table_with_choppers_full_beamline_range(wavelength_from):
    wf = _make_workflow(wavelength_from)
    wf[unwrap.DiskChoppers[AnyRun]] = _make_choppers()
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 0], unit='m')
    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 64
    wf[unwrap.PulseStride[AnyRun]] = 1

    wf[unwrap.LtotalRange[AnyRun, snx.NXdetector]] = (
        sc.scalar(5.0, unit='m'),
        sc.scalar(65.0, unit='m'),
    )
    wf[unwrap.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[unwrap.TimeResolution] = sc.scalar(250.0, unit='us')

    table = wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])

    # Close to source: early times and large spread
    da = table.array['distance', 2]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() >= sc.scalar(0.0, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(1.0e3, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(2.0e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(3.0e4, unit="us").to(unit=eto.unit)

    # Just after WFM choppers, very small range
    da = table.array['distance', 27]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(4.0e3, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(5.0e3, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(7.0e3, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(8.0e3, unit="us").to(unit=eto.unit)

    # Before last chopper
    da = table.array['distance', 272]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(9.0e3, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(1.2e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(3.1e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(3.5e4, unit="us").to(unit=eto.unit)

    # At the top of the table
    da = table.array['distance', -3]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(1.7e4, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(2.2e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(6.4e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(6.9e4, unit="us").to(unit=eto.unit)


@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_raises_for_distance_before_source(wavelength_from):
    wf = _make_workflow(wavelength_from)
    wf[unwrap.DiskChoppers[AnyRun]] = {}
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 10], unit='m')
    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 65
    wf[unwrap.PulseStride[AnyRun]] = 1

    # Setting the starting point at zero will make a table that would cover a range
    # from -0.2m to 65.0m
    wf[unwrap.LtotalRange[AnyRun, snx.NXdetector]] = (
        sc.scalar(0.0, unit='m'),
        sc.scalar(65.0, unit='m'),
    )
    wf[unwrap.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[unwrap.TimeResolution] = sc.scalar(250.0, unit='us')

    with pytest.raises(ValueError, match="Building the lookup table failed"):
        _ = wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])


@pytest.mark.parametrize("detector_or_monitor", ["detector", "monitor"])
@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_workflow_computes_table_using_alias(detector_or_monitor, wavelength_from):
    # LookupTableWorkflow is an old (deprecated) alias for GenericUnwrapWorkflow
    wf = LookupTableWorkflow(use_simulation=(wavelength_from == "simulation"))
    wf[unwrap.DiskChoppers[AnyRun]] = {}
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, 0], unit='m')
    wf[unwrap.PulseStride[AnyRun]] = 1

    if wavelength_from == "simulation":
        wf[unwrap.NumberOfSimulatedNeutrons] = 100_000
        wf[unwrap.SimulationSeed] = 60

    lmin, lmax = sc.scalar(25.0, unit='m'), sc.scalar(35.0, unit='m')
    dres = sc.scalar(0.1, unit='m')
    tres = sc.scalar(333.0, unit='us')

    Comp = snx.NXdetector if detector_or_monitor == "detector" else FrameMonitor0

    wf[unwrap.LtotalRange[AnyRun, Comp]] = lmin, lmax
    wf[unwrap.DistanceResolution] = dres
    wf[unwrap.TimeResolution] = tres

    table = wf.compute(unwrap.LookupTable[AnyRun, Comp])

    assert table.array.coords['distance'].min() < lmin
    assert table.array.coords['distance'].max() > lmax
    assert table.array.coords['event_time_offset'].max() == sc.scalar(
        1 / 14, unit='s'
    ).to(unit=table.array.coords['event_time_offset'].unit)
    assert sc.isclose(table.distance_resolution, dres)
    # Note that the time resolution is not exactly preserved since we want the table to
    # span exactly the frame period.
    assert sc.isclose(table.time_resolution, tres, rtol=sc.scalar(0.01))


def test_lut_workflow_guesses_pulse_stride():
    wf = _make_workflow()
    choppers = _make_choppers()
    wf[unwrap.DiskChoppers[AnyRun]] = choppers

    for i in range(1, 4):
        choppers["pulse-skipping"] = DiskChopper(
            axle_position=sc.vector([0, 0, 20], unit='m'),
            frequency=sc.scalar(-14 / i, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-10, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[0.0], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[120.0], unit='deg'),
            slit_height=None,
            radius=None,
        )
        wf[unwrap.DiskChoppers[AnyRun]] = choppers

        assert wf.compute(unwrap.PulseStride[AnyRun]) == i


@pytest.mark.parametrize("wavelength_from", ["analytical", "simulation"])
def test_lut_does_not_raise_if_no_neutrons_make_it_through(wavelength_from):
    wf = _make_workflow(wavelength_from)
    # Add a very slowly rotating chopper that will block all neutrons.
    wf[unwrap.DiskChoppers[AnyRun]] = {
        'chopper1': DiskChopper(
            axle_position=sc.vector([0, 0, -15.0], unit='m'),
            frequency=sc.scalar(0.1, unit='Hz'),
            beam_position=sc.scalar(0.0, unit='deg'),
            phase=sc.scalar(0.0, unit='rad'),
            slit_begin=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            slit_end=sc.array(dims=['cutout'], values=[90.0], unit='deg'),
            slit_height=None,
            radius=sc.scalar(0.35, unit='m'),
        )
    }
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, -25.0], unit='m')
    # Need to force the pulse stride so that it doesn't get set to a large value due to
    # the slow chopper.
    wf[unwrap.PulseStride[AnyRun]] = 1
    wf[unwrap.LtotalRange[AnyRun, snx.NXdetector]] = (
        sc.scalar(15.0, unit='m'),
        sc.scalar(30.0, unit='m'),
    )
    wf[unwrap.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[unwrap.TimeResolution] = sc.scalar(250.0, unit='us')

    table = wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])

    # Chopper is 10m from the source, LUT starts at 15m, so no neutrons should make it
    # through. The table should be full of NaNs but should not raise an error.
    assert sc.all(sc.isnan(table.array.data))


def test_analytical_lut_does_not_raise_with_degenerate_polygon():
    wf = _make_workflow("analytical")
    # This chopper is open slightly before t=0 and closes exactly at t=0. Only
    # unphysical wavelengths of 0 (infinite speed) could go through. A degenerate
    # polygon (all vertices are 0) is created from the chopper cascade, and we need to
    # ensure that this does not cause the LUT computation to fail. It should just drop
    # this polygon and return NaNs for the corresponding wavelengths.
    wf[unwrap.DiskChoppers[AnyRun]] = {
        'chopper1': DiskChopper(
            axle_position=sc.vector([0, 0, -15.0], unit='m'),
            frequency=sc.scalar(14.0, unit='Hz'),
            beam_position=sc.scalar(0.0, unit='deg'),
            phase=sc.scalar(0.0, unit='rad'),
            slit_begin=sc.array(dims=['cutout'], values=[0.0], unit='deg'),
            slit_end=sc.array(dims=['cutout'], values=[90.0], unit='deg'),
            slit_height=None,
            radius=sc.scalar(0.35, unit='m'),
        )
    }
    wf[SourceBounds] = SourceBounds(
        time=(sc.scalar(0.0, unit='ms'), sc.scalar(5.0, unit='ms')),
        wavelength=(
            sc.scalar(0.0, unit='angstrom'),  #  Min wavelength to 0
            sc.scalar(15.0, unit='angstrom'),
        ),
    )
    wf[Position[snx.NXsource, AnyRun]] = sc.vector([0, 0, -25.0], unit='m')
    wf[unwrap.PulseStride[AnyRun]] = 1
    wf[unwrap.LtotalRange[AnyRun, snx.NXdetector]] = (
        sc.scalar(15.0, unit='m'),
        sc.scalar(30.0, unit='m'),
    )
    wf[unwrap.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[unwrap.TimeResolution] = sc.scalar(250.0, unit='us')

    table = wf.compute(unwrap.LookupTable[AnyRun, snx.NXdetector])

    # Chopper is 10m from the source, LUT starts at 15m, so no neutrons should make it
    # through. The table should be full of NaNs but should not raise an error.
    assert sc.all(sc.isnan(table.array.data))
