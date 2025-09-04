import numpy as np
import scipp as sc
import scippneutron as scn
from scipp.testing import assert_allclose

from ess.beer import BeerModMcStasWorkflow, BeerModMcStasWorkflowKnownPeaks
from ess.beer.data import duplex_peaks_array, mcstas_duplex
from ess.beer.types import DHKLList
from ess.reduce.nexus.types import Filename, SampleRun
from ess.reduce.time_of_flight.types import DetectorTofData


def test_can_reduce_using_known_peaks_workflow():
    wf = BeerModMcStasWorkflowKnownPeaks()
    wf[DHKLList] = duplex_peaks_array()
    wf[Filename[SampleRun]] = mcstas_duplex(7)
    da = wf.compute(DetectorTofData[SampleRun])
    assert 'bank1' in da
    assert 'bank2' in da
    da = da['bank1']
    assert 'tof' in da.coords
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
    da = wf.compute(DetectorTofData[SampleRun])
    assert 'bank1' in da
    assert 'bank2' in da
    da = da['bank1']
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
