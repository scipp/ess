# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import h5py
import numpy as np
import pytest
import sciline
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    EmptyDetector,
    Filename,
    GravityVector,
    NeXusName,
    Position,
    RawDetector,
    TimeInterval,
)
from ess.reduce.unwrap.types import DetectorLtotal, ErrorLimitedLookupTable
from scipp.testing import assert_allclose, assert_identical

from ess import freia
from ess.freia import conversions
from ess.freia.mirror import (
    mirror_geometry,
    reflected_upstream_position,
)
from ess.freia.types import (
    DetectorRegionOfInterest,
    DownstreamSlitCenters,
    MirrorGeometry,
    MirrorNormal,
    MirrorPosition,
    SampleSurfaceNormal,
    ShutterPositions,
    ShutterStatusLogs,
    UpstreamSlitCenters,
)
from ess.reflectometry.types import (
    CoordTransformationGraph,
    CorrectedDetector,
    QBins,
    ReducibleData,
    ReferenceRun,
    ReflectivityOverQ,
    RunType,
    Sample,
    SampleRun,
    WavelengthDetector,
)


def _set_mirror(workflow, run=SampleRun):
    workflow[MirrorPosition[run]] = sc.vector([0.0, 0.0, -1.0], unit='m')
    workflow[MirrorNormal[run]] = sc.vector([0.0, 2.0, 0.0])


@pytest.mark.parametrize(
    'factory',
    [getattr(freia, name) for name in freia.__all__ if name.endswith('Workflow')],
)
def test_all_workflow_constructors_accept_inversion(factory):
    point = sc.vector([0.0, 2.0, -3.0], unit='m')
    ordinary = factory()
    assert ordinary.compute(MirrorGeometry[SampleRun] | None) is None
    inverted = factory(inversion=True)
    _set_mirror(inverted)
    assert_identical(
        reflected_upstream_position(
            point,
            sc.vector([0.0, 1.0, 0.0], unit='m'),
            inverted.compute(MirrorGeometry[SampleRun] | None),
        ),
        sc.vector([0.0, -2.0, -3.0], unit='m'),
    )


@pytest.mark.parametrize('normal_scale', [0.5, -2.0, 3.0])
def test_reflected_path_through_tilted_translated_mirror(normal_scale):
    rotation = sc.spatial.rotations_from_rotvecs(
        sc.vector([0.3, -0.4, 0.1], unit='rad')
    )
    position = sc.vector([0.1, 0.2, -0.3], unit='m')
    normal = rotation * sc.vector([0.0, 1.0, 0.0])
    # A 45-degree ray hits the mirror at position. Rotate the whole setup.
    upstream = position + rotation * sc.vector([0.0, 2.0, -2.0], unit='m')
    downstream = position + rotation * sc.vector([0.0, 0.5, 0.5], unit='m')
    reflected = reflected_upstream_position(
        upstream,
        downstream.to(unit='cm'),
        mirror_geometry(position.to(unit='mm'), normal_scale * normal),
    )
    expected = position + rotation * sc.vector([0.0, -2.0, -2.0], unit='m')
    assert_allclose(reflected, expected)
    assert_allclose(
        sc.norm(downstream - reflected), sc.scalar(2.5 * np.sqrt(2), unit='m')
    )


@pytest.mark.parametrize('normal', [[0, 0, 0], [0, np.nan, 0], [0, np.inf, 0]])
def test_invalid_mirror_normal_is_rejected(normal):
    with pytest.raises(ValueError, match='MirrorNormal'):
        mirror_geometry(sc.vector([0.0, 0.0, 0.0], unit='m'), sc.vector(normal))


@pytest.mark.parametrize('side', [-1, 1])
def test_reflected_upstream_position_supports_either_mirror_side_and_mixed_units(side):
    mirror = mirror_geometry(
        sc.vector([0.0, 0.0, -1.0], unit='m'), sc.vector([0.0, 1.0, 0.0])
    )
    assert_allclose(
        reflected_upstream_position(
            sc.vector([0.0, side * 2000.0, -3000.0], unit='mm'),
            sc.vector([0.0, side * 0.5, -0.5], unit='m'),
            mirror,
        ),
        sc.vector([0.0, -side * 2000.0, -3000.0], unit='mm'),
    )


def test_ordinary_path_preserves_positions_exactly():
    position = sc.vector([0.0, 1234.56789, -3456.78912], unit='mm')
    assert_identical(
        reflected_upstream_position(position, sc.vector([0.0, 0.0, 0.0], unit='m')),
        position,
    )


def test_custom_conversion_pipeline_defaults_to_no_mirror():
    wf = sciline.Pipeline(conversions.providers, constraints={RunType: [SampleRun]})
    wf[Position[snx.NXsource, SampleRun]] = sc.vector([0.0, 0.0, -2.0], unit='m')
    wf[Position[snx.NXsample, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[SampleSurfaceNormal[SampleRun]] = sc.vector([0.0, 1.0, 0.0])
    wf[GravityVector] = sc.vector([0.0, -9.80665, 0.0], unit='m/s^2')
    graph = wf.compute(CoordTransformationGraph[SampleRun])
    result = sc.DataArray(sc.scalar(0.0)).transform_coords('L1', graph=graph)
    assert_allclose(result.coords['L1'], sc.scalar(2.0, unit='m'))


def _write_mirror(path, *, name='inversion_mirror', angles=(80.0,)):
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = 'NXinstrument'
        mirror = instrument.create_group(name)
        mirror.attrs['NX_class'] = 'NXmirror'
        mirror['depends_on'] = 'transformations/pitch'
        transforms = mirror.create_group('transformations')
        transforms.attrs['NX_class'] = 'NXtransformations'
        z = transforms.create_dataset('distance', data=-1.0)
        z.attrs.update(
            transformation_type='translation',
            vector=[0.0, 0.0, 1.0],
            units='m',
            depends_on='.',
        )
        y = transforms.create_dataset('height', data=-300.0)
        y.attrs.update(
            transformation_type='translation',
            vector=[0.0, 1.0, 0.0],
            units='mm',
            depends_on='distance',
        )
        pitch = transforms.create_group('pitch')
        pitch.attrs.update(
            NX_class='NXlog',
            transformation_type='rotation',
            vector=[1.0, 0.0, 0.0],
            depends_on='height',
        )
        pitch['value'] = angles
        pitch['value'].attrs['units'] = 'deg'
        pitch['time'] = np.arange(len(angles), dtype=np.int64)
        pitch['time'].attrs.update(units='s', start='2026-01-01T00:00:00Z')


@pytest.mark.parametrize('name', ['inversion_mirror', 'deflector'])
@pytest.mark.parametrize('run', [SampleRun, ReferenceRun])
def test_nexus_mirror_loads_translation_and_local_minus_z_normal(tmp_path, name, run):
    filename = tmp_path / 'mirror.nxs'
    _write_mirror(filename, name=name)
    wf = freia.FreiaWorkflow(inversion=True)
    wf[Filename[run]] = filename
    wf[NeXusName[snx.NXmirror]] = name
    assert_allclose(
        wf.compute(MirrorPosition[run]), sc.vector([0.0, -0.3, -1.0], unit='m')
    )
    angle = np.deg2rad(80.0)
    assert_allclose(
        wf.compute(MirrorNormal[run]), sc.vector([0.0, np.sin(angle), -np.cos(angle)])
    )
    wf.compute(MirrorGeometry[run] | None)


def test_moving_mirror_requires_a_selected_pose(tmp_path):
    filename = tmp_path / 'mirror.nxs'
    _write_mirror(filename, angles=(80.0, 81.0))
    wf = freia.FreiaWorkflow(inversion=True)
    wf[Filename[SampleRun]] = filename
    with pytest.raises(ValueError, match='time-dependent'):
        wf.compute(MirrorNormal[SampleRun])
    wf[TimeInterval[SampleRun]] = TimeInterval(slice(0, 1))
    assert_allclose(
        wf.compute(MirrorNormal[SampleRun]),
        sc.vector([0.0, np.sin(np.deg2rad(80)), -np.cos(np.deg2rad(80))]),
    )


def test_missing_mirror_fails_with_manual_geometry_instructions(tmp_path):
    filename = tmp_path / 'mirror.nxs'
    _write_mirror(filename, name='another_mirror')
    wf = freia.FreiaWorkflow(inversion=True)
    wf[Filename[SampleRun]] = filename
    with pytest.raises(ValueError, match='supply MirrorPosition and MirrorNormal'):
        wf.compute(MirrorGeometry[SampleRun] | None)
    _set_mirror(wf)
    wf.compute(MirrorGeometry[SampleRun] | None)


def _set_run(workflow, run, *, direct=False, supply_wavelength=True):
    _set_mirror(workflow, run)
    workflow[GravityVector] = sc.vector([0.0, 0.0, 0.0], unit='m/s^2')
    workflow[Position[snx.NXsource, run]] = sc.vector([0.0, 3.0, -4.0], unit='m')
    workflow[Position[snx.NXsample, run]] = sc.vector([0.0, 1.0, 0.0], unit='m')
    workflow[SampleSurfaceNormal[run]] = sc.vector([0.0, -1.0, 0.0])
    workflow[DetectorRegionOfInterest[run]] = {}
    # Incoming downward 45-degree ray reflects at (0,0,-1), hits (0,1,0),
    # and reflects down onto the detector. The direct beam continues upwards.
    data = sc.DataArray(
        sc.array(
            dims=['event'],
            values=[100.0 if direct else 20.0],
            variances=[100.0 if direct else 20.0],
            unit='counts',
        ),
        coords={
            'wavelength': sc.array(dims=['event'], values=[4.0], unit='angstrom'),
            'pixel_id': sc.array(dims=['event'], values=[0], unit=None),
        },
    ).group('pixel_id')
    data.coords['position'] = sc.vectors(
        dims=['pixel_id'], values=[[0.0, 2.0 if direct else 0.0, 1.0]], unit='m'
    )
    if supply_wavelength:
        workflow[WavelengthDetector[run]] = data
    workflow[EmptyDetector[run]] = data.bins.sum()
    return data


@pytest.mark.parametrize('run', [SampleRun, ReferenceRun])
def test_detector_flight_path_includes_mirror_detour(run):
    wf = freia.FreiaWorkflow(inversion=True)
    _set_run(wf, run, direct=run == ReferenceRun, supply_wavelength=False)
    assert_allclose(
        wf.compute(DetectorLtotal[run]),
        sc.array(dims=['pixel_id'], values=[5 * np.sqrt(2)], unit='m'),
    )
    # The angular graph must use the same incident path through the mirror.
    graph = wf.compute(CoordTransformationGraph[run])
    result = sc.DataArray(sc.scalar(0.0)).transform_coords('L1', graph=graph)
    assert_allclose(result.coords['L1'], sc.scalar(4 * np.sqrt(2), unit='m'))


def test_mirror_before_sample_rejects_source_on_opposite_side_of_plane():
    wf = freia.FreiaWorkflow(inversion=True)
    _set_run(wf, SampleRun, supply_wavelength=False)
    wf[Position[snx.NXsource, SampleRun]] = sc.vector([0.0, -3.0, -4.0], unit='m')
    with pytest.raises(ValueError, match='same side of the mirror plane'):
        wf.compute(DetectorLtotal[SampleRun])
    graph = wf.compute(CoordTransformationGraph[SampleRun])
    with pytest.raises(ValueError, match='same side of the mirror plane'):
        sc.DataArray(sc.scalar(0.0)).transform_coords('L1', graph=graph)


def test_slits_on_opposite_sides_of_mirror_plane_are_rejected():
    wf = freia.FreiaOffspecWorkflow(inversion=True, incident_beam_from='fixed_slits')
    _set_run(wf, SampleRun)
    wf[UpstreamSlitCenters[SampleRun]] = {
        'beam': sc.vector([0.0, -2.0, -3.0], unit='m')
    }
    wf[DownstreamSlitCenters[SampleRun]] = {
        'beam': sc.vector([0.0, 0.5, -0.5], unit='m')
    }
    with pytest.raises(ValueError, match='same side of the mirror plane'):
        wf.compute(CorrectedDetector[SampleRun])


def test_wavelength_reconstruction_uses_reflected_flight_path():
    wf = freia.FreiaWorkflow(inversion=True)
    data = _set_run(wf, SampleRun, supply_wavelength=False)
    wavelength = sc.scalar(4.0, unit='angstrom')
    # Independent sum of the three physical segments: 3 sqrt(2), sqrt(2), sqrt(2).
    distance = sc.scalar(5 * np.sqrt(2), unit='m')
    arrival = (distance * sc.constants.m_n * wavelength / sc.constants.h).to(unit='s')
    wf[RawDetector[SampleRun]] = data.bins.drop_coords('wavelength').bins.assign_coords(
        event_time_offset=sc.bins_like(data, arrival)
    )
    distances = sc.linspace('distance', 5.0, 10.0, 2, unit='m')
    times = sc.linspace('event_time_offset', 0.0, 0.01, 2, unit='s')
    # Ideal instantaneous source, with no choppers or frame overlap.
    wf[ErrorLimitedLookupTable[SampleRun, snx.NXdetector]] = ErrorLimitedLookupTable(
        array=sc.DataArray(
            (sc.constants.h / sc.constants.m_n / distances * times).to(unit='angstrom'),
            coords={'distance': distances, 'event_time_offset': times},
        ),
        pulse_period=sc.scalar(1 / 14, unit='s'),
        pulse_stride=1,
        distance_resolution=sc.scalar(5.0, unit='m'),
        time_resolution=sc.scalar(0.01, unit='s'),
    )
    result = wf.compute(WavelengthDetector[SampleRun])
    assert_allclose(
        result.bins.coords['wavelength'].bins.constituents['data'].squeeze(), wavelength
    )
    assert_allclose(result.coords['Ltotal'].squeeze(), distance)


def test_inverted_specular_sample_and_direct_beam_normalization():
    wf = freia.FreiaUnnormalizedWorkflow(inversion=True)
    for run in (SampleRun, ReferenceRun):
        _set_run(wf, run, direct=run == ReferenceRun)
    wf[QBins] = sc.array(dims=['Q'], values=[2.0, 2.5], unit='1/angstrom')
    wf[Sample] = wf[ReducibleData[SampleRun]]  # Skip footprint correction.
    result = wf.compute(ReflectivityOverQ)
    assert_allclose(result.data, sc.array(dims=['Q'], values=[0.2], variances=[0.0024]))
    sample = wf.compute(CorrectedDetector[SampleRun])
    assert_allclose(
        sample.bins.coords['theta'].bins.constituents['data'],
        sc.array(dims=['event'], values=[np.pi / 4], unit='rad'),
    )
    assert_allclose(
        sample.bins.coords['scattering_angle'].bins.constituents['data'],
        sc.array(dims=['event'], values=[-np.pi / 4], unit='rad'),
    )


@pytest.mark.parametrize('method', ['closest', 'fixed_slits', 'shutter_logs'])
def test_offspecular_inversion_reflects_slit_angles_and_shutter_paths(method):
    wf = freia.FreiaOffspecWorkflow(inversion=True, incident_beam_from=method)
    data = _set_run(wf, SampleRun)
    wf[UpstreamSlitCenters[SampleRun]] = {'beam': sc.vector([0.0, 2.0, -3.0], unit='m')}
    wf[DownstreamSlitCenters[SampleRun]] = {
        'beam': sc.vector([0.0, 0.5, -0.5], unit='m')
    }
    if method == 'shutter_logs':
        # Shutter -> mirror -> sample -> detector is 4 sqrt(2) m. Place a
        # closing edge between that arrival time and the incorrect direct path.
        wf[ShutterPositions[SampleRun]] = {
            'beam': sc.vector([0.0, 2.0, -3.0], unit='m')
        }
        speed = (
            (sc.constants.h / sc.constants.m_n / sc.scalar(4.0, unit='angstrom'))
            .to(unit='m/s')
            .value
        )
        arrival = 4 * np.sqrt(2) / speed
        constituents = data.bins.constituents
        content = constituents['data']
        content.coords['event_time_zero'] = sc.datetimes(
            dims=['event'], values=['2026-01-01T00:00:00'], unit='ns'
        )
        content.coords['event_time_offset'] = sc.array(
            dims=['event'], values=[arrival], unit='s'
        )
        data = sc.DataArray(sc.bins(**constituents), coords=data.coords)
        wf[ShutterStatusLogs[SampleRun]] = {
            'beam': sc.DataArray(
                sc.array(dims=['time'], values=[True, False]),
                coords={
                    'time': sc.datetimes(
                        dims=['time'],
                        values=['2026-01-01T00:00:00', '2026-01-01T00:00:00.0005'],
                        unit='ns',
                    )
                },
            )
        }
    wf[WavelengthDetector[SampleRun]] = data
    result = wf.compute(CorrectedDetector[SampleRun])
    angles = result.coords.get(
        'incident_angle', result.bins.coords.get('incident_angle')
    )
    if angles.bins is not None:
        angles = angles.bins.constituents['data'].squeeze()
    assert_allclose(angles, sc.scalar(np.pi / 4, unit='rad'))
    assert_allclose(
        result.bins.coords['Qx'].bins.constituents['data'],
        sc.array(dims=['event'], values=[0.0], unit='1/angstrom'),
        atol=sc.scalar(1e-15, unit='1/angstrom'),
    )
    assert_allclose(
        result.bins.coords['Qz'].bins.constituents['data'],
        sc.array(dims=['event'], values=[np.pi / np.sqrt(2)], unit='1/angstrom'),
    )
    assert result.sum().value == 20.0


def test_gravity_and_physical_positions_remain_in_laboratory_frame():
    wf = freia.FreiaWorkflow(inversion=True)
    _set_run(wf, SampleRun)
    gravity = sc.vector([0.0, -9.80665, 0.0], unit='m/s^2')
    wf[GravityVector] = gravity
    graph = wf.compute(CoordTransformationGraph[SampleRun])
    assert_identical(graph['gravity'](), gravity)
    assert_identical(
        graph['sample_position'](), wf.compute(Position[snx.NXsample, SampleRun])
    )
    assert_identical(
        wf.compute(Position[snx.NXsource, SampleRun]),
        sc.vector([0.0, 3.0, -4.0], unit='m'),
    )
