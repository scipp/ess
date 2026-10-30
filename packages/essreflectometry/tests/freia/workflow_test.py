# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from ess.reduce.nexus.types import GravityVector, Position
from ess.reduce.unwrap.types import KeepEventTimeOffset
from scipp.testing import assert_allclose, assert_identical
from scippnexus import NXsample, NXsource

from ess.freia import FreiaOffspecWorkflow, FreiaWorkflow
from ess.freia.corrections import RunNormalization
from ess.freia.types import (
    DetectorRegionOfInterest,
    DownstreamSlitCenters,
    SampleSurfaceNormal,
    ShutterPositions,
    ShutterStatusLogs,
    UpstreamSlitCenters,
    WavelengthMonitor,
)
from ess.reflectometry.corrections import footprint_on_sample
from ess.reflectometry.types import (
    BeamSize,
    CoordTransformationGraph,
    CorrectedDetector,
    QBins,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    Sample,
    SampleRun,
    SampleSize,
    WavelengthBins,
    WavelengthDetector,
)


def _make_workflow(run_norm):
    wf = FreiaWorkflow(run_norm=run_norm)
    wf[GravityVector] = sc.vector([0.0, 0.0, 0.0], unit='m/s^2')
    for run, sign, weights in [
        (SampleRun, 1, [20.0, 40.0, 999.0]),
        (ReferenceRun, -1, [100.0, 100.0, 999.0]),
    ]:
        events = sc.DataArray(
            sc.array(dims=['event'], values=weights, variances=weights, unit='counts'),
            coords={
                'wavelength': sc.array(
                    dims=['event'], values=[2.0, 4.0, 2.0], unit='angstrom'
                ),
                'pixel_id': sc.array(dims=['event'], values=[0, 0, 1], unit=None),
            },
        ).group('pixel_id')
        # Sample rotation is 10 degrees. The matching beams make angles of
        # 30 degrees with the surface; the second pixel is outside the ROI.
        angles = np.deg2rad([10.0 + sign * 30.0, 10.0 + sign * 60.0])
        events.coords['position'] = sc.vectors(
            dims=['pixel_id'],
            values=[[0, np.sin(angle), np.cos(angle)] for angle in angles],
            unit='m',
        )
        wf[WavelengthDetector[run]] = events
        wf[Position[NXsample, run]] = sc.vector([0.0, 0.0, 0.0], unit='m')
        # Source position must not set the angular reference frame.
        wf[Position[NXsource, run]] = sc.vector([3.0, 2.0, -20.0], unit='m')
        wf[SampleSurfaceNormal[run]] = sc.vector(
            [0.0, np.cos(np.deg2rad(10.0)), -np.sin(np.deg2rad(10.0))]
        )
        low, high = sorted([10.0 + sign * 20.0, 10.0 + sign * 40.0])
        wf[DetectorRegionOfInterest[run]] = {
            'scattering_angle': (
                sc.scalar(low, unit='deg'),
                sc.scalar(high, unit='deg'),
            )
        }
    wf[WavelengthBins] = sc.array(
        dims=['wavelength'], values=[1.0, 3.0, 5.0], unit='angstrom'
    )
    wf[QBins] = sc.array(dims=['Q'], values=[1.0, 2.0, 4.0, 6.0], unit='1/angstrom')
    return wf


def test_reduce_reflectivity_with_monitor_and_footprint():
    wf = _make_workflow(RunNormalization.monitor_histogram)
    sample_size = sc.scalar(10.0, unit='mm')
    beam_size = sc.scalar(5.0, unit='mm')
    wf[SampleSize[SampleRun]] = sample_size
    wf[BeamSize[SampleRun]] = beam_size
    for run, values in [(SampleRun, [4.0, 8.0]), (ReferenceRun, [2.0, 2.0])]:
        wf[WavelengthMonitor[run]] = sc.DataArray(
            sc.array(dims=['wavelength'], values=values, unit='counts'),
            coords={'wavelength': wf.compute(WavelengthBins)},
        )

    results = wf.compute(
        (ReflectivityOverQ, ReducibleData[ReferenceRun], CorrectedDetector[SampleRun])
    )
    result = results[ReflectivityOverQ]
    direct_beam = results[ReducibleData[ReferenceRun]]
    assert 'theta' not in direct_beam.bins.coords
    assert 'Q' not in direct_beam.bins.coords
    for data in (direct_beam, results[CorrectedDetector[SampleRun]]):
        coords = set(data.coords) | set(data.bins.coords)
        assert not {'incident_beam', 'scattered_beam', 'L1', 'L2'} & coords
        assert 'scattering_angle' in coords
    assert 'outgoing_direction' in direct_beam.bins.coords
    assert 'sample_surface_normal' in direct_beam.coords
    sample = results[CorrectedDetector[SampleRun]]
    assert {'theta', 'Q'} <= set(sample.bins.coords)
    assert not {'outgoing_direction', 'sample_surface_normal'} & (
        set(sample.coords) | set(sample.bins.coords)
    )

    fraction = footprint_on_sample(sc.scalar(30.0, unit='deg'), beam_size, sample_size)
    # Q bins contain wavelengths 4 and 2 angstrom, respectively.
    expected = (
        sc.array(dims=['Q'], values=[0.1, 0.1], variances=[0.00035, 0.0006]) / fraction
    )
    assert_allclose(result['Q', :2].data, expected)
    assert_identical(
        result.masks['direct_beam'],
        sc.array(dims=['Q'], values=[False, False, True]),
    )


def test_rebinning_integrates_before_dividing():
    wf = _make_workflow(RunNormalization.none)
    wf[Sample] = wf[ReducibleData[SampleRun]]
    wf[QBins] = sc.array(dims=['Q'], values=[1.0, 4.0], unit='1/angstrom')

    result = wf.compute(ReflectivityOverQ)

    assert_allclose(
        result.data,
        sc.array(
            dims=['Q'], values=[60.0 / 200.0], variances=[0.3**2 * (1 / 60 + 1 / 200)]
        ),
    )


def test_offspec_workflow_provides_qx_and_qz():
    workflow = FreiaOffspecWorkflow()
    workflow[Position[NXsample, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    workflow[Position[NXsource, SampleRun]] = sc.vector([0.0, 0.0, -10.0], unit='m')
    workflow[SampleSurfaceNormal[SampleRun]] = sc.vector([0.0, 1.0, 0.0])
    workflow[GravityVector] = sc.vector([0.0, 0.0, 0.0], unit='m/s^2')
    workflow[UpstreamSlitCenters[SampleRun]] = {
        key: sc.vector([0.0, 0.0, -2.0], unit='m')
        for key in ('top', 'middle', 'bottom')
    }
    workflow[DownstreamSlitCenters[SampleRun]] = {
        key: sc.vector([0.0, y, -1.0], unit='m')
        for key, y in zip(('top', 'middle', 'bottom'), (-0.1, -0.2, -0.3), strict=True)
    }

    graph = workflow.compute(CoordTransformationGraph[SampleRun])

    assert {'incident_angle', 'reflection_angle', 'Qx', 'Qz'} <= graph.keys()
    assert 'hypothetical_incident_direction' not in graph
    assert 'incident_direction' not in graph


@pytest.mark.parametrize('method', ['closest', 'fixed_slits'])
def test_offspec_workflow_supports_scalar_incidence_for_a_single_beam(method):
    source = _make_workflow(RunNormalization.none)
    wf = FreiaOffspecWorkflow(incident_beam_from=method)
    assert wf.compute(KeepEventTimeOffset) is False
    for key in (
        WavelengthDetector[SampleRun],
        Position[NXsample, SampleRun],
        Position[NXsource, SampleRun],
        SampleSurfaceNormal[SampleRun],
        GravityVector,
    ):
        wf[key] = source.compute(key)
    wf[UpstreamSlitCenters[SampleRun]] = {'beam': sc.vector([0.0, 0.0, -2.0], unit='m')}
    wf[DownstreamSlitCenters[SampleRun]] = {
        'beam': sc.vector([0.0, 0.0, -1.0], unit='m')
    }

    result = wf.compute(CorrectedDetector[SampleRun])

    assert_allclose(
        result.coords['incident_angle'], sc.scalar(10.0, unit='deg').to(unit='rad')
    )
    reflection_angles = np.deg2rad([30.0, 30.0, 60.0])
    wavenumbers = 2 * np.pi / np.array([2.0, 4.0, 2.0])
    assert_allclose(
        result.bins.coords['Qx'].bins.constituents['data'],
        sc.array(
            dims=['event'],
            values=wavenumbers * (np.cos(reflection_angles) - np.cos(np.deg2rad(10.0))),
            unit='1/angstrom',
        ),
    )
    assert_allclose(
        result.bins.coords['Qz'].bins.constituents['data'],
        sc.array(
            dims=['event'],
            values=wavenumbers * (np.sin(reflection_angles) + np.sin(np.deg2rad(10.0))),
            unit='1/angstrom',
        ),
    )
    assert_identical(
        result.bins.data, source.compute(WavelengthDetector[SampleRun]).bins.data
    )


@pytest.mark.parametrize(
    ('method', 'expected'),
    [('closest', [1.0, 1.0, 1.0, 1.0]), ('shutter_logs', [0.3, 1.0, np.nan, np.nan])],
)
def test_offspec_workflow_assigns_incidence_with_independent_methods(method, expected):
    wf = FreiaOffspecWorkflow(incident_beam_from=method)
    wf[GravityVector] = sc.vector([0.0, 0.0, 0.0], unit='m/s^2')
    wf[Position[NXsample, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[Position[NXsource, SampleRun]] = sc.vector([0.0, 0.0, -29.0], unit='m')
    wf[SampleSurfaceNormal[SampleRun]] = sc.vector([0.0, 1.0, 0.0])
    angles = {'a': 0.3, 'b': 1.0}
    wf[UpstreamSlitCenters[SampleRun]] = {
        name: sc.vector([0.0, 2 * np.tan(np.deg2rad(angle)), -2.0], unit='m')
        for name, angle in angles.items()
    }
    wf[DownstreamSlitCenters[SampleRun]] = {
        name: sc.vector([0.0, np.tan(np.deg2rad(angle)), -1.0], unit='m')
        for name, angle in angles.items()
    }
    # All four reflected rays look like the 1 degree beam to the closest method.
    data = (
        sc.DataArray(
            sc.ones(dims=['event'], shape=[4], with_variances=True, unit='counts'),
            coords={
                'pixel': sc.zeros(dims=['event'], shape=[4], dtype='int64', unit=None),
                'event_time_zero': sc.datetimes(
                    dims=['event'], values=[0] * 4, unit='ns'
                ),
                'event_time_offset': sc.array(
                    dims=['event'], values=[30, 40, 50, 60], unit='ms'
                ),
                'wavelength': (
                    sc.constants.h
                    / sc.constants.m_n
                    / sc.full(dims=['event'], shape=[4], value=1000.0, unit='m/s')
                ).to(unit='angstrom'),
            },
        )
        .group('pixel')
        .assign_coords(
            position=sc.vectors(
                dims=['pixel'],
                values=[[0.0, np.sin(np.deg2rad(1.0)), np.cos(np.deg2rad(1.0))]],
                unit='m',
            ),
            Ltotal=sc.scalar(30.0, unit='m'),
        )
    )
    wf[WavelengthDetector[SampleRun]] = data

    def log(times, states):
        return sc.DataArray(
            sc.array(dims=['time'], values=states),
            coords={
                'time': sc.datetimes(dims=['time'], values=times, unit='ms'),
            },
        )

    if method == 'shutter_logs':
        # Shutters are independent of the slit geometry and at different distances
        # from the source. Reverse their key order to check pairing by beam name.
        wf[ShutterPositions[SampleRun]] = {
            'b': sc.vector([0.0, 0.0, -9.0], unit='m'),
            'a': sc.vector([0.0, 0.0, -19.0], unit='m'),
        }
        wf[ShutterStatusLogs[SampleRun]] = {
            'a': log([0, 15, 25, 35], [True, False, True, False]),
            'b': log([0, 25, 45], [False, True, False]),
        }

    result = wf.compute(CorrectedDetector[SampleRun])
    assigned = result.bins.coords['incident_angle'].bins.constituents['data']
    assert_allclose(
        assigned,
        sc.array(dims=['event'], values=expected, unit='deg').to(unit='rad'),
        equal_nan=True,
    )
    assert {'scattering_angle', 'reflection_angle', 'Qx', 'Qz'} <= set(
        result.bins.coords
    )
    assert not {
        'incident_beam',
        'scattered_beam',
        'outgoing_direction',
        'sample_surface_normal',
        'L1',
        'L2',
        'theta',
        'Q',
    } & (set(result.coords) | set(result.bins.coords))
    assert not any(name.startswith('time_at_shutter') for name in result.bins.coords)
    assert_identical(result.bins.data, data.bins.data)
    assert_identical(
        result.bins.masks['incident_beam_unassigned'].bins.constituents['data'],
        sc.array(
            dims=['event'],
            values=[False, False, method == 'shutter_logs', method == 'shutter_logs'],
        ),
    )
    assert result.sum().value == (2.0 if method == 'shutter_logs' else 4.0)
