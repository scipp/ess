# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    NeXusComponent,
    NeXusTransformation,
    NeXusTransformationChain,
    TimeInterval,
    TransformationTimeFilter,
)
from scipp.testing import assert_allclose

from ess.estia import EstiaWorkflow
from ess.estia.load import load_detector_rotation, load_sample_rotation
from ess.reflectometry.types import DetectorRotation, RawSampleRotation, SampleRun


def _group(parent, name, nx_class):
    g = parent.create_group(name)
    g.attrs['NX_class'] = nx_class
    return g


def _rotation(parent, name, angles, depends_on='.', vector=(0.0, 1.0, 0.0)):
    g = _group(parent, name, 'NXlog')
    g.attrs.update(transformation_type='rotation', vector=vector, depends_on=depends_on)
    angles = np.atleast_1d(angles)
    g.create_dataset('value', data=angles).attrs['units'] = 'degrees'
    t = g.create_dataset('time', data=np.arange(len(angles), dtype='int64') * 10)
    t.attrs.update(units='s', start='2026-11-04T00:00:00Z')
    return g


def _workflow(tmp_path, nx_class, angles, *, entry_name='entry'):
    """Load a small real NeXus component into the production ESTIA pipeline."""
    path = tmp_path / 'rotations.nxs'
    with h5py.File(path, 'w') as f:
        entry = _group(f, entry_name, 'NXentry')
        inst = _group(entry, 'instrument', 'NXinstrument')
        if nx_class is snx.NXsample:
            component = _group(entry, 'sample', 'NXsample')
        else:
            component = _group(inst, 'detector', 'NXdetector')
            component.create_dataset('detector_number', data=[1])
            component.create_dataset('data', data=[1.0]).attrs['units'] = 'counts'
        # Arbitrary names ensure that no individual motor is looked up by name.
        stage = _group(inst, 'stage', 'NXpositioner')
        rotation = _rotation(stage, 'value', angles)
        tip = _rotation(inst, 'additional_rotation', 0.25, rotation.name)
        translation = inst.create_dataset('translation', data=123.0)
        translation.attrs.update(
            transformation_type='translation',
            vector=[1.0, 0.0, 0.0],
            depends_on=tip.name,
            units='mm',
        )
        component['depends_on'] = translation.name
        component_path = component.name
    with snx.File(path) as f:
        loaded = f[component_path][()]
    wf = EstiaWorkflow()
    wf[NeXusComponent[nx_class, SampleRun]] = loaded
    return wf


@pytest.mark.parametrize('entry_name', ['entry', 'another_entry'])
@pytest.mark.parametrize(
    ('nx_class', 'result_type'),
    [(snx.NXsample, RawSampleRotation), (snx.NXdetector, DetectorRotation)],
)
def test_orientation_from_nexus(tmp_path, entry_name, nx_class, result_type):
    wf = _workflow(tmp_path, nx_class, 3.65, entry_name=entry_name)
    assert_allclose(wf.compute(result_type[SampleRun]), sc.scalar(3.9, unit='deg'))


@pytest.mark.parametrize(
    ('nx_class', 'result_type'),
    [(snx.NXsample, RawSampleRotation), (snx.NXdetector, DetectorRotation)],
)
def test_orientation_uses_selected_time_interval(tmp_path, nx_class, result_type):
    wf = _workflow(tmp_path, nx_class, [1.0, 2.0, 3.0])
    # NXlog values apply until the following entry: 2 degrees applies at t=15 s.
    wf[TimeInterval[SampleRun]] = TimeInterval(
        slice(
            sc.datetime('2026-11-04T00:00:15', unit='s'),
            sc.datetime('2026-11-04T00:00:16', unit='s'),
        )
    )
    assert_allclose(wf.compute(result_type[SampleRun]), sc.scalar(2.25, unit='deg'))


@pytest.mark.parametrize(
    ('nx_class', 'result_type'),
    [(snx.NXsample, RawSampleRotation), (snx.NXdetector, DetectorRotation)],
)
def test_orientation_uses_configured_time_filter(tmp_path, nx_class, result_type):
    wf = _workflow(tmp_path, nx_class, [1.0, 2.0, 3.0])
    wf[TransformationTimeFilter[nx_class, SampleRun]] = lambda log: log.mean()
    assert_allclose(wf.compute(result_type[SampleRun]), sc.scalar(2.25, unit='deg'))


def test_sample_rejects_unselected_time_dependent_orientation(tmp_path):
    wf = _workflow(tmp_path, snx.NXsample, [1.0, 2.0])
    with pytest.raises(ValueError, match='time-dependent'):
        wf.compute(RawSampleRotation[SampleRun])


@pytest.mark.parametrize('loader', [load_sample_rotation, load_detector_rotation])
def test_orientation_rejects_retained_time_dependence(loader):
    transform = sc.DataArray(
        sc.spatial.rotations_from_rotvecs(
            sc.vectors(dims=['time'], values=[[0, 1, 0], [0, 2, 0]], unit='deg')
        ),
        coords={
            'time': sc.datetimes(
                dims=['time'], values=['2026-11-04T00:00:00', '2026-11-04T00:00:10']
            )
        },
    )
    with pytest.raises(ValueError, match='static orientation'):
        loader(NeXusTransformation(transform))


@pytest.mark.parametrize(
    ('nx_class', 'result_type', 'expected'),
    [
        (snx.NXsample, RawSampleRotation, -25.3150511771),
        (snx.NXdetector, DetectorRotation, 1.25),
    ],
)
def test_orientation_composes_rotations_about_different_axes(
    tmp_path, nx_class, result_type, expected
):
    wf = _workflow(tmp_path, nx_class, 1.0)
    chain = wf.compute(NeXusTransformationChain[nx_class, SampleRun])
    tip = chain.absolute_path()
    for name, axis, angle in [('pitch', [1, 0, 0], 30.0), ('roll', [0, 0, 1], 45.0)]:
        path = f'/entry/instrument/{name}'
        chain.transformations[path] = snx.nxtransformations.Transform(
            name=path,
            transformation_type='rotation',
            value=sc.scalar(angle, unit='deg'),
            vector=sc.vector(axis),
            depends_on=snx.DependsOn(parent='/entry/instrument', value=tip),
        )
        tip = path
    chain.value = tip
    wf[NeXusTransformationChain[nx_class, SampleRun]] = chain
    # For the sample surface, yaw = 1.25 - atan(sin(30) * tan(45)).
    # The detector +z axis is unchanged by the innermost roll about z.
    assert_allclose(wf.compute(result_type[SampleRun]), sc.scalar(expected, unit='deg'))
