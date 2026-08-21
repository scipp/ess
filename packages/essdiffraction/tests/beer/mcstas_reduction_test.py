import importlib
import sys

import numpy as np
import pytest
import scipp as sc
from ess.beer import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
    BeerPowderMcStasWorkflow,
)
from ess.beer.data import (
    mcstas_duplex,
    mcstas_few_neutrons_3d_detector_example,
    mcstas_more_neutrons_3d_detector_example,
    mcstas_powder_silicon_in_vanadium_can,
    mcstas_silicon_new_model,
    silicon_peaks_array,
)
from ess.beer.mcstas import (
    load_beer_mcstas,
    load_beer_mcstas_monitor,
)
from ess.beer.types import DetectorBank, DHKLList, WavelengthDetector
from ess.powder.types import (
    DspacingDetector,
    ElasticCoordTransformGraph,
    SampleRun,
)
from scipp.testing import assert_allclose

from ess.reduce.nexus.types import Filename

_DSPACE_BINS = sc.linspace('dspacing', 0.8, 2.2, 4001, unit='angstrom')


def test_can_reduce_using_known_peaks_workflow():
    wf = BeerModMcStasWorkflowKnownPeaks()
    wf[DHKLList] = silicon_peaks_array()
    wf[DetectorBank] = DetectorBank.north
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(7)
    result = wf.compute(
        (WavelengthDetector[SampleRun], ElasticCoordTransformGraph[SampleRun])
    )
    da = result[WavelengthDetector[SampleRun]]
    assert 'wavelength' in da.bins.coords
    # assert dataarray has all coords required to compute dspacing
    da = da.transform_coords(
        ('dspacing',),
        graph=result[ElasticCoordTransformGraph[SampleRun]],
    )
    h = da.hist(dspacing=_DSPACE_BINS, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(5e-4, unit='angstrom'),
    )


@pytest.mark.parametrize(
    'fname',
    [
        mcstas_silicon_new_model(7),
        mcstas_silicon_new_model(10),
        mcstas_silicon_new_model(16),
        mcstas_more_neutrons_3d_detector_example(),
    ],
)
def test_can_reduce_using_unknown_peaks_workflow(fname):
    wf = BeerModMcStasWorkflow()
    wf[Filename[SampleRun]] = fname
    wf[DetectorBank] = DetectorBank.north
    result = wf.compute(
        (WavelengthDetector[SampleRun], ElasticCoordTransformGraph[SampleRun])
    )
    da = result[WavelengthDetector[SampleRun]]
    assert 'wavelength' in da.bins.coords
    da = da.transform_coords(
        ('dspacing',),
        graph=result[ElasticCoordTransformGraph[SampleRun]],
    )
    h = da.hist(dspacing=_DSPACE_BINS, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        # The two peaks around 1.6 are very similar in magnitude,
        # so either of them can be bigger and that is fine.
        sc.scalar(1.5677, unit='angstrom')
        if max_peak_d < sc.scalar(1.6, unit='angstrom')
        else sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(5e-4, unit='angstrom'),
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
    h = da.hist(dspacing=_DSPACE_BINS, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(5e-4, unit='angstrom'),
    )


def test_powder_mcstas_analytical_workflow_computes_dspacing():
    wf = BeerPowderMcStasWorkflow()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(6)
    wf[DetectorBank] = DetectorBank.north

    da = wf.compute(DspacingDetector[SampleRun])

    assert 'wavelength' in da.bins.coords
    assert 'dspacing' in da.bins.coords
    h = da.hist(dspacing=_DSPACE_BINS, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(5e-4, unit='angstrom'),
    )


@pytest.mark.parametrize(
    'fname',
    [
        pytest.param(mcstas_duplex(7), id='legacy-2d'),
        pytest.param(mcstas_silicon_new_model(7), id='new-2d'),
        pytest.param(mcstas_few_neutrons_3d_detector_example(), id='panelized-3d'),
        pytest.param(mcstas_powder_silicon_in_vanadium_can(), id='powder-2d'),
    ],
)
@pytest.mark.parametrize('bank', DetectorBank)
def test_can_load_all_detector_generations(fname, bank):
    da = load_beer_mcstas(fname, bank)

    assert da.coords['pixel_id'].dtype == sc.DType.int32
    assert 'position' in da.coords
    assert 'event_time_offset' in da.bins.coords
    assert da.bins.size().sum().value > 0


def test_load_both_detector_banks():
    filename = mcstas_few_neutrons_3d_detector_example()
    north = load_beer_mcstas(filename, DetectorBank.north)
    south = load_beer_mcstas(filename, DetectorBank.south)
    both = load_beer_mcstas(filename, DetectorBank.both)

    assert both.bins.size().sum().value == (
        north.bins.size().sum().value + south.bins.size().sum().value
    )


def test_loaded_mcstas_event_variances_are_squared_weights():
    da = load_beer_mcstas(mcstas_few_neutrons_3d_detector_example(), DetectorBank.north)
    weights = da.bins.constituents['data']

    assert weights.variances is not None
    assert_allclose(sc.variances(weights), sc.values(weights) ** 2)


def test_can_load_monitor():
    da = load_beer_mcstas_monitor(mcstas_few_neutrons_3d_detector_example())
    assert 'wavelength' in da.coords
    assert 'position' in da.coords
    assert da.coords['position'].dtype == sc.DType.vector3
    assert da.coords['position'].unit == 'm'


def test_io_module_reexports_mcstas_loaders():
    sys.modules.pop('ess.beer.io', None)

    with pytest.warns(DeprecationWarning, match='ess.beer.io'):
        io = importlib.import_module('ess.beer.io')

    assert io.load_beer_mcstas is load_beer_mcstas
