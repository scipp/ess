# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    NeXusComponent,
    TimeInterval,
    TransformationTimeFilter,
)
from scipp.testing import assert_allclose

from ess.estia import EstiaWorkflow
from ess.reflectometry.types import DetectorRotation, RawSampleRotation, SampleRun


def _group(parent, name, nx_class):
    g = parent.create_group(name)
    g.attrs['NX_class'] = nx_class
    return g


def _rotation(parent, name, angles, depends_on='.'):
    g = _group(parent, name, 'NXlog')
    g.attrs.update(
        transformation_type='rotation', vector=[0.0, 1.0, 0.0], depends_on=depends_on
    )
    angles = np.atleast_1d(angles)
    g.create_dataset('value', data=angles).attrs['units'] = 'degrees'
    t = g.create_dataset('time', data=np.arange(len(angles), dtype='int64') * 10)
    t.attrs.update(units='s', start='2026-11-04T00:00:00Z')
    return g


def _workflow(tmp_path, nx_class, angles, *, additional_rotation, entry_name='entry'):
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
        tip = _rotation(inst, 'additional_rotation', additional_rotation, rotation.name)
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
def test_orientation_combines_y_rotations(tmp_path, entry_name, nx_class, result_type):
    wf = _workflow(
        tmp_path, nx_class, 3.65, additional_rotation=0.25, entry_name=entry_name
    )
    assert_allclose(wf.compute(result_type[SampleRun]), sc.scalar(3.9, unit='deg'))


@pytest.mark.parametrize(
    ('nx_class', 'result_type'),
    [(snx.NXsample, RawSampleRotation), (snx.NXdetector, DetectorRotation)],
)
def test_orientation_uses_selected_time_interval(tmp_path, nx_class, result_type):
    # The selected value must differ from the detector's default whole-log mean.
    wf = _workflow(tmp_path, nx_class, [1.0, 2.0, 7.0], additional_rotation=0.25)
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
    wf = _workflow(tmp_path, nx_class, [1.0, 2.0, 3.0], additional_rotation=0.25)
    # Selecting the last value distinguishes the override from the default mean.
    wf[TransformationTimeFilter[nx_class, SampleRun]] = lambda log: log['time', -1]
    assert_allclose(wf.compute(result_type[SampleRun]), sc.scalar(3.25, unit='deg'))
