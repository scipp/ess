import numpy as np
import pytest
import scipp as sc
import scippneutron as scn
from ess.beer import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
)
from ess.beer.data import (
    duplex_peaks_array,
    mcstas_duplex,
    mcstas_few_neutrons_3d_detector_example,
    mcstas_more_neutrons_3d_detector_example,
    mcstas_silicon_new_model,
)
from ess.beer.io import (
    load_beer_mcstas,
    load_beer_mcstas_monitor,
    mcstas_chopper_delay_from_mode_new_simulations,
)
from ess.beer.types import DetectorBank, DHKLList, WavelengthDetector
from ess.powder.types import ElasticCoordTransformGraph, RawDetector, SampleRun
from scipp.testing import assert_allclose

from ess.reduce.nexus.types import DetectorBankSizes, Filename


def test_can_reduce_using_known_peaks_workflow():
    wf = BeerModMcStasWorkflowKnownPeaks()
    wf[DHKLList] = duplex_peaks_array()
    wf[DetectorBank] = DetectorBank.north
    wf[Filename[SampleRun]] = mcstas_duplex(7)
    da = wf.compute(WavelengthDetector[SampleRun])
    assert 'wavelength' in da.bins.coords
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


@pytest.mark.parametrize(
    'fname', [mcstas_silicon_new_model(7), mcstas_more_neutrons_3d_detector_example()]
)
def test_can_reduce_using_unknown_peaks_workflow(fname):
    wf = BeerModMcStasWorkflow()
    wf[Filename[SampleRun]] = fname
    wf[DetectorBank] = DetectorBank.north
    wf.insert(mcstas_chopper_delay_from_mode_new_simulations)
    da = wf.compute(WavelengthDetector[SampleRun])
    da = da.transform_coords(
        ('dspacing',),
        graph=scn.conversion.graph.tof.elastic('tof'),
    )
    h = da.hist(dspacing=2000, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        # The two peaks around 1.6 are very similar in magnitude,
        # so either of them can be bigger and that is fine.
        sc.scalar(1.5677, unit='angstrom')
        if max_peak_d < sc.scalar(1.6, unit='angstrom')
        else sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(1e-2, unit='angstrom'),
    )


def test_pulse_shaping_workflow():
    wf = BeerMcStasWorkflowPulseShaping()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(6)
    wf[DetectorBank] = DetectorBank.north
    res = wf.compute(
        (WavelengthDetector[SampleRun], ElasticCoordTransformGraph[SampleRun])
    )
    da = res[WavelengthDetector[SampleRun]]
    assert 'wavelength' in da.bins.coords
    # assert dataarray has all coords required to compute dspacing
    da = da.transform_coords(
        ('dspacing',),
        graph=res[ElasticCoordTransformGraph[SampleRun]],
    )
    h = da.hist(dspacing=2000, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(1e-2, unit='angstrom'),
    )


def test_can_load_3d_detector():
    sizes = {'north_detector': {'x': 10, 'y': 20}, 'south_detector': {'x': 10, 'y': 20}}
    load_beer_mcstas(
        mcstas_few_neutrons_3d_detector_example(), DetectorBank.north, sizes
    )
    da = load_beer_mcstas(
        mcstas_few_neutrons_3d_detector_example(), DetectorBank.south, sizes
    )
    assert 'panel' in da.dims


def test_can_load_monitor():
    da = load_beer_mcstas_monitor(mcstas_few_neutrons_3d_detector_example())
    assert 'wavelength' in da.coords
    assert 'position' in da.coords
    assert da.coords['position'].dtype == sc.DType.vector3
    assert da.coords['position'].unit == 'm'


@pytest.mark.parametrize(
    ('bank_in_sizes', 'bank'),
    [
        ('south_detector', DetectorBank.south),
        ('north_detector', DetectorBank.north),
        (DetectorBank.south, DetectorBank.south),
        (DetectorBank.north, DetectorBank.north),
    ],
)
@pytest.mark.parametrize(
    'fname', [mcstas_silicon_new_model(7), mcstas_more_neutrons_3d_detector_example()]
)
def test_detector_bank_size_parameter_determines_loaded_detector_size(
    bank_in_sizes, bank, fname
):
    wf = BeerMcStasWorkflowPulseShaping()
    wf[Filename[SampleRun]] = fname
    wf[DetectorBank] = bank
    wf[DetectorBankSizes] = {bank_in_sizes: {'x': 10, 'y': 20}}
    res = wf.compute(RawDetector[SampleRun])
    assert res.sizes['x'] == 10
    assert res.sizes['y'] == 20
