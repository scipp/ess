import numpy as np
import scipp as sc
import scippneutron as scn
from scipp.testing import assert_allclose

from ess.beer import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
)
from ess.beer.data import duplex_peaks_array, mcstas_duplex, mcstas_silicon_new_model
from ess.beer.types import DetectorBank, DHKLList
from ess.reduce.nexus.types import Filename, SampleRun
from ess.reduce.time_of_flight.types import TofDetector


def test_can_reduce_using_known_peaks_workflow():
    wf = BeerModMcStasWorkflowKnownPeaks()
    wf[DHKLList] = duplex_peaks_array()
    wf[DetectorBank] = 1
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
    wf[DetectorBank] = 1
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
    wf[DetectorBank] = 1
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
