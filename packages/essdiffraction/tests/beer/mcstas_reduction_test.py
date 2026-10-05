import importlib
import re
import sys

import mcstastox
import numpy as np
import pytest
import sciline as sl
import scipp as sc
import scippnexus as snx
from ess.beer import (
    BeerMcStasWorkflowPulseShaping,
    BeerModMcStasWorkflow,
    BeerModMcStasWorkflowKnownPeaks,
    BeerModulationAutoMcStasWorkflow,
    BeerModulationKnownPeaksMcStasWorkflow,
    BeerPowderMcStasWorkflow,
    BeerPowderWorkflow,
    BeerPowderWorkflowAnalytical,
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
    mcstas_providers,
)
from ess.beer.mcstas.beamline import (
    ModulationMode,
    PulseShapingMode,
    simulation_choppers,
)
from ess.beer.types import DetectorBank, DHKLList, WavelengthDetector
from ess.powder.types import (
    DspacingBins,
    DspacingDetector,
    DspacingNBins,
    ElasticCoordTransformGraph,
    KeepEvents,
    MaskedDetectorIDs,
    NormalizedDspacing,
    QDetector,
    SampleRun,
    TofMask,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    WavelengthMask,
)
from scipp.testing import assert_allclose

from ess.reduce.nexus.types import DiskChoppers, Filename, Position
from ess.reduce.unwrap import PulseStride

_DSPACE_BINS = sc.linspace('dspacing', 0.8, 2.2, 4001, unit='angstrom')


@pytest.mark.parametrize(
    ('factory', 'replacement'),
    [
        (BeerModMcStasWorkflow, 'BeerModulationAutoMcStasWorkflow'),
        (BeerModMcStasWorkflowKnownPeaks, 'BeerModulationKnownPeaksMcStasWorkflow'),
        (BeerMcStasWorkflowPulseShaping, 'BeerPowderMcStasWorkflow'),
        (
            BeerPowderWorkflowAnalytical,
            "BeerPowderWorkflow(wavelength_from='analytical')",
        ),
    ],
)
def test_deprecated_workflow_names_warn_with_replacement(factory, replacement):
    with pytest.warns(DeprecationWarning, match=re.escape(replacement)):
        workflow = factory()

    assert isinstance(workflow, sl.Pipeline)


def test_can_reduce_using_known_peaks_workflow():
    wf = BeerModulationKnownPeaksMcStasWorkflow()
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
    wf = BeerModulationAutoMcStasWorkflow()
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


@pytest.mark.parametrize(
    'factory',
    [BeerModulationAutoMcStasWorkflow, BeerModulationKnownPeaksMcStasWorkflow],
)
def test_modulation_workflows_can_normalize(factory):
    wf = factory()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(7)
    wf[DetectorBank] = DetectorBank.north
    wf[DHKLList] = silicon_peaks_array()
    wf[DspacingBins] = sc.linspace('dspacing', 0.8, 2.2, 31, unit='angstrom')
    wf[MaskedDetectorIDs] = MaskedDetectorIDs({})
    wf[KeepEvents[SampleRun]] = KeepEvents[SampleRun](True)
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    wf[TofMask] = None
    wf[WavelengthMask] = None
    wf[TwoThetaMask] = None

    result = wf.compute(NormalizedDspacing[SampleRun])

    assert result.bins.size().sum().value > 0


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


def _beer_powder_mcstas_workflow(wavelength_from):
    wf = BeerPowderWorkflow(wavelength_from=wavelength_from)
    for provider in mcstas_providers:
        wf.insert(provider)
    return wf


@pytest.mark.parametrize(
    'pulse_skipping', [False, True], ids=['normal', 'pulse-skipping']
)
@pytest.mark.parametrize(
    ('make_workflow', 'mode'),
    [
        pytest.param(
            lambda: _beer_powder_mcstas_workflow('analytical'),
            6,
            id='powder-analytical',
        ),
        pytest.param(
            lambda: _beer_powder_mcstas_workflow('simulation'),
            6,
            id='powder-simulation',
        ),
        pytest.param(
            lambda: _beer_powder_mcstas_workflow('file'),
            6,
            id='powder-file',
        ),
        pytest.param(BeerModulationAutoMcStasWorkflow, 7, id='modulation'),
        pytest.param(
            BeerModulationKnownPeaksMcStasWorkflow, 7, id='modulation-known-peaks'
        ),
        pytest.param(BeerPowderMcStasWorkflow, 6, id='powder-mcstas'),
    ],
)
def test_beer_workflows_compute_dspacing_bins_without_loading_events(
    monkeypatch, make_workflow, mode, pulse_skipping
):
    wf = make_workflow()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(mode)
    wf[DetectorBank] = DetectorBank.north
    wf[DHKLList] = silicon_peaks_array()
    wf[DspacingNBins] = 123

    def fail_if_events_are_loaded(*args, **kwargs):
        raise AssertionError('event data must not be used to determine bin edges')

    monkeypatch.setattr(mcstastox.Read, 'get_event_data', fail_if_events_are_loaded)

    if pulse_skipping:
        # Use pulse-skipping choppers with the existing detector geometry.
        source_position = wf.compute(Position[snx.NXsource, SampleRun])
        chopper_mode = ModulationMode.ds0 if mode == 7 else PulseShapingMode.ds1
        wf[DiskChoppers[SampleRun]] = simulation_choppers(chopper_mode, source_position)

    bins = wf.compute(DspacingBins)

    assert wf.compute(PulseStride[SampleRun]) == (2 if pulse_skipping else 1)
    assert bins.sizes == {'dspacing': 124}
    assert sc.all(sc.isfinite(bins)).value
    assert sc.all(bins[1:] > bins[:-1]).value


def test_powder_mcstas_analytical_workflow_computes_q():
    wf = BeerPowderMcStasWorkflow()
    wf[Filename[SampleRun]] = mcstas_silicon_new_model(6)
    wf[DetectorBank] = DetectorBank.north

    da = wf.compute(QDetector[SampleRun])

    assert 'Q' in da.bins.coords


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
