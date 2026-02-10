import numpy as np
import scipp as sc
import scippneutron as scn
from scipp.testing import assert_allclose

from ess.beer import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
)
from ess.beer.data import (
    duplex_peaks_array,
    mcstas_duplex,
    mcstas_few_neutrons_3d_detector_example,
    mcstas_silicon_new_model,
)
from ess.beer.io import load_beer_mcstas, load_beer_mcstas_monitor
from ess.beer.types import DetectorBank, DHKLList
from ess.reduce.nexus.types import Filename, SampleRun
from ess.reduce.time_of_flight.types import TofDetector


def test_can_reduce_using_known_peaks_workflow():
    wf = BeerModMcStasWorkflowKnownPeaks()
    wf[DHKLList] = duplex_peaks_array()
    wf[DetectorBank] = DetectorBank.north
    wf[Filename[SampleRun]] = mcstas_duplex(7)
    da = wf.compute(TofDetector[SampleRun])
    assert 'tof' in da.bins.coords
    # assert dataarray has all coords required to compute dspacing
    da = da.transform_coords(
        ('dspacing',),
        graph=scn.conversion.graph.tof.elastic('tof'),
    )
    h = da.hist(dspacing=2000, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(2.0407, unit='angstrom'),
        atol=sc.scalar(1e-2, unit='angstrom'),
    )


def test_can_reduce_using_unknown_peaks_workflow():
    wf = BeerModMcStasWorkflow()
    wf[Filename[SampleRun]] = mcstas_duplex(7)
    wf[DetectorBank] = DetectorBank.north
    da = wf.compute(TofDetector[SampleRun])
    da = da.transform_coords(
        ('dspacing',),
        graph=scn.conversion.graph.tof.elastic('tof'),
    )
    h = da.hist(dspacing=2000, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(2.0407, unit='angstrom'),
        atol=sc.scalar(1e-2, unit='angstrom'),
    )


def test_pulse_shaping_workflow():
    wf = BeerMcStasWorkflowPulseShaping()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(6)
    wf[DetectorBank] = DetectorBank.north
    da = wf.compute(TofDetector[SampleRun])
    assert 'tof' in da.bins.coords
    # assert dataarray has all coords required to compute dspacing
    da = da.transform_coords(
        ('dspacing',),
        graph=scn.conversion.graph.tof.elastic('tof'),
    )
    h = da.hist(dspacing=2000, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(1e-2, unit='angstrom'),
    )


def test_can_load_3d_detector():
    load_beer_mcstas(mcstas_few_neutrons_3d_detector_example(), DetectorBank.north)
    da = load_beer_mcstas(mcstas_few_neutrons_3d_detector_example(), DetectorBank.south)
    assert 'panel' in da.dims


def test_can_load_monitor():
    da = load_beer_mcstas_monitor(mcstas_few_neutrons_3d_detector_example())
    assert 'wavelength' in da.coords
    assert 'position' in da.coords
    assert da.coords['position'].dtype == sc.DType.vector3
    assert da.coords['position'].unit == 'm'
