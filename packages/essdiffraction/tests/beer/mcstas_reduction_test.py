import importlib
import sys

import numpy as np
import pytest
import scipp as sc
import scippneutron as scn
from ess.beer import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
    BeerPowderMcStasWorkflowAnalytical,
)
from ess.beer.data import (
    duplex_peaks_array,
    mcstas_duplex,
    mcstas_few_neutrons_3d_detector_example,
    mcstas_more_neutrons_3d_detector_example,
    mcstas_silicon_new_model,
)
from ess.beer.mcstas import (
    load_beer_mcstas,
    load_beer_mcstas_monitor,
    mcstas_chopper_delay_from_mode_new_simulations,
)
from ess.beer.types import DetectorBank, DHKLList, WavelengthDetector
from ess.powder.types import (
    DspacingDetector,
    ElasticCoordTransformGraph,
    IntensityDspacingTwoTheta,
    MaskedDetectorIDs,
    PixelMaskFilename,
    RawDetector,
    SampleRun,
)
from scipp.testing import assert_allclose

from ess.reduce import workflow as reduce_workflow
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
        atol=sc.scalar(2e-3, unit='angstrom'),
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
        atol=sc.scalar(2e-3, unit='angstrom'),
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
        atol=sc.scalar(2e-3, unit='angstrom'),
    )


def test_powder_mcstas_analytical_workflow_computes_dspacing():
    wf = BeerPowderMcStasWorkflowAnalytical()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(6)
    wf[DetectorBank] = DetectorBank.north

    da = wf.compute(DspacingDetector[SampleRun])

    assert 'wavelength' in da.bins.coords
    assert 'dspacing' in da.bins.coords
    h = da.hist(dspacing=2000, dim=da.dims)
    max_peak_d = sc.midpoints(h['dspacing', np.argmax(h.values)].coords['dspacing'])[0]
    assert_allclose(
        max_peak_d,
        sc.scalar(1.6374, unit='angstrom'),
        atol=sc.scalar(2e-3, unit='angstrom'),
    )


def test_powder_mcstas_analytical_workflow_exposes_pixel_mask_parameter():
    wf = BeerPowderMcStasWorkflowAnalytical()
    spec = reduce_workflow.workflow_registry.get(BeerPowderMcStasWorkflowAnalytical)

    specs = reduce_workflow.get_parameters(
        wf, (IntensityDspacingTwoTheta[SampleRun],), spec.parameters
    )
    wf = reduce_workflow.assign_parameter_values(wf, {PixelMaskFilename: ()}, specs)

    assert PixelMaskFilename in specs
    assert wf.compute(MaskedDetectorIDs) == {}


def test_can_load_3d_detector():
    sizes = {
        'north_detector': {'x': 500, 'y': 200},
        'south_detector': {'x': 500, 'y': 200},
    }
    load_beer_mcstas(
        mcstas_few_neutrons_3d_detector_example(), DetectorBank.north, sizes
    )
    da = load_beer_mcstas(
        mcstas_few_neutrons_3d_detector_example(), DetectorBank.south, sizes
    )
    assert 'panel' in da.dims
    # Detector position.x is monotone in panel dimension.
    panel_x_diff = np.diff(da.coords['detector_position'].fields.x.values)
    assert (panel_x_diff > 0).all() or (panel_x_diff < 0).all()


def test_load_pulse_shaping_detector_adds_nominal_time_at_chopper():
    sizes = {
        'north_detector': {'x': 500, 'y': 200},
        'south_detector': {'x': 500, 'y': 200},
    }

    da = load_beer_mcstas(mcstas_silicon_new_model(6), DetectorBank.north, sizes)

    assert 'wavelength_estimate' not in da.coords
    assert_allclose(
        da.coords['nominal_time_at_chopper'].to(unit='ms'),
        sc.scalar(5.07692, unit='ms'),
        atol=sc.scalar(1e-5, unit='ms'),
    )


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
