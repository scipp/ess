# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess.reduce.nexus import GenericNeXusWorkflow, patch
from ess.reduce.nexus.types import (
    EmptyDetector,
    Filename,
    NeXusDetectorName,
    NeXusFilePatch,
    PreopenNeXusFile,
    SampleRun,
)

DETECTOR = '/entry/instrument/detector'


@pytest.fixture
def nexus_file(tmp_path: Path) -> Path:
    path = tmp_path / 'file.nxs'
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        entry['title'] = 'original title'
        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = 'NXinstrument'
        detector = instrument.create_group('detector')
        detector.attrs['NX_class'] = 'NXdetector'
        detector.attrs['description'] = 'a detector'
        detector['detector_number'] = np.arange(1, 5)
        detector['data'] = np.ones(4)
        for name in ('x_pixel_offset', 'y_pixel_offset', 'z_pixel_offset'):
            detector.create_dataset(name, data=np.zeros(4)).attrs['units'] = 'm'
        detector['depends_on'] = f'{DETECTOR}/transformations/offset'
        transformations = detector.create_group('transformations')
        transformations.attrs['NX_class'] = 'NXtransformations'
        offset = transformations.create_dataset('offset', data=2.0)
        offset.attrs['transformation_type'] = 'translation'
        offset.attrs['vector'] = [0.0, 0.0, 1.0]
        offset.attrs['units'] = 'm'
        offset.attrs['depends_on'] = '.'
        entry['number_link'] = h5py.SoftLink(f'{DETECTOR}/detector_number')
    return path


def open_patched(path: Path, ops: list[patch.PatchOp]) -> patch.PatchedNeXusFile:
    return patch.open_patched_file(path, ops, definitions=snx.base_definitions())


def test_empty_patch_loads_same_as_unpatched_file(nexus_file: Path) -> None:
    with snx.File(nexus_file) as f:
        expected = f[DETECTOR][()]
    with open_patched(nexus_file, []) as f:
        sc.testing.assert_identical(f[DETECTOR][()], expected)


def test_patch_does_not_modify_file(nexus_file: Path) -> None:
    ops = [
        patch.Remove(path='/entry/title'),
        patch.SetAttrs(path=DETECTOR, attrs={'description': 'patched'}),
    ]
    with open_patched(nexus_file, ops) as f:
        assert 'title' not in f['entry']
    with h5py.File(nexus_file) as f:
        assert f['entry/title'][()] == b'original title'
        assert f[DETECTOR].attrs['description'] == 'a detector'


def test_set_attrs_adds_and_replaces_attributes(nexus_file: Path) -> None:
    ops = [
        patch.SetAttrs(path=DETECTOR, attrs={'description': 'patched', 'new': 1.5}),
    ]
    with open_patched(nexus_file, ops) as f:
        attrs = f[DETECTOR].attrs
        assert attrs['description'] == 'patched'
        assert attrs['new'] == 1.5
        assert attrs['NX_class'] == 'NXdetector'


def test_set_attrs_on_dataset(nexus_file: Path) -> None:
    ops = [patch.SetAttrs(path='/entry/title', attrs={'units': 'm'})]
    with open_patched(nexus_file, ops) as f:
        assert f['entry/title'].attrs['units'] == 'm'


def test_set_attrs_raises_if_path_does_not_exist(nexus_file: Path) -> None:
    with pytest.raises(KeyError, match='/entry/missing'):
        open_patched(nexus_file, [patch.SetAttrs(path='/entry/missing', attrs={})])


def test_patch_paths_must_be_absolute(nexus_file: Path) -> None:
    with pytest.raises(ValueError, match='absolute'):
        open_patched(nexus_file, [patch.Remove(path='entry/title')])


def test_patch_paths_are_normalized(nexus_file: Path) -> None:
    ops = [
        patch.SetAttrs(path=f'/{DETECTOR}/', attrs={'x': 1}),
        patch.SetAttrs(path=f'{DETECTOR}/../detector', attrs={'y': 2}),
    ]
    with open_patched(nexus_file, ops) as f:
        assert f[DETECTOR].attrs['x'] == 1
        assert f[DETECTOR].attrs['y'] == 2


def test_cannot_remove_root(nexus_file: Path) -> None:
    with pytest.raises(ValueError, match='root'):
        open_patched(nexus_file, [patch.Remove(path='/')])


def test_delete_attrs(nexus_file: Path) -> None:
    ops = [
        patch.SetAttrs(path=DETECTOR, attrs={'new': 1}),
        patch.DeleteAttrs(path=DETECTOR, names=['description', 'new']),
    ]
    with open_patched(nexus_file, ops) as f:
        assert set(f[DETECTOR].attrs) == {'NX_class'}


def test_delete_attrs_raises_if_attribute_does_not_exist(nexus_file: Path) -> None:
    with pytest.raises(KeyError, match='missing'):
        open_patched(nexus_file, [patch.DeleteAttrs(path=DETECTOR, names=['missing'])])


def test_remove_group(nexus_file: Path) -> None:
    ops = [patch.Remove(path=f'{DETECTOR}/transformations')]
    with open_patched(nexus_file, ops) as f:
        assert 'transformations' not in f[DETECTOR]
        with pytest.raises(KeyError):
            f[f'{DETECTOR}/transformations/offset']


def test_remove_raises_if_path_does_not_exist(nexus_file: Path) -> None:
    ops = [patch.Remove(path='/entry/title'), patch.Remove(path='/entry/title')]
    with pytest.raises(KeyError, match='/entry/title'):
        open_patched(nexus_file, ops)


def test_move_renames_group(nexus_file: Path) -> None:
    ops = [patch.Move(source=DETECTOR, target='/entry/instrument/renamed')]
    with open_patched(nexus_file, ops) as f:
        instrument = f['entry/instrument']
        assert list(instrument) == ['renamed']
        renamed = instrument['renamed']
        assert renamed.name == '/entry/instrument/renamed'
        assert renamed['transformations/offset'].name == (
            '/entry/instrument/renamed/transformations/offset'
        )


def test_move_keeps_patches_of_moved_objects(nexus_file: Path) -> None:
    ops = [
        patch.SetAttrs(path=f'{DETECTOR}/transformations', attrs={'x': 1}),
        patch.Remove(path=f'{DETECTOR}/detector_number'),
        patch.Move(source=DETECTOR, target='/entry/detector'),
    ]
    with open_patched(nexus_file, ops) as f:
        assert f['entry/detector/transformations'].attrs['x'] == 1
        assert 'detector_number' not in f['entry/detector']


def test_move_raises_if_target_exists(nexus_file: Path) -> None:
    ops = [patch.Move(source='/entry/title', target='/entry/instrument')]
    with pytest.raises(ValueError, match='exists'):
        open_patched(nexus_file, ops)


def test_move_onto_removed_path(nexus_file: Path) -> None:
    ops = [
        patch.Remove(path='/entry/title'),
        patch.Move(source=f'{DETECTOR}/depends_on', target='/entry/title'),
    ]
    with open_patched(nexus_file, ops) as f:
        assert f['entry/title'][()] == f'{DETECTOR}/transformations/offset'
        assert 'depends_on' not in f[DETECTOR]


def test_set_dataset_adds_dataset(nexus_file: Path) -> None:
    ops = [
        patch.SetDataset(path='/entry/new', value=[1.0, 2.0]),
        patch.SetAttrs(path='/entry/new', attrs={'units': 'mm'}),
    ]
    with open_patched(nexus_file, ops) as f:
        sc.testing.assert_identical(
            f['entry/new'][()], sc.array(dims=['dim_0'], values=[1.0, 2.0], unit='mm')
        )


def test_set_dataset_replaces_string_dataset(nexus_file: Path) -> None:
    ops = [patch.SetDataset(path='/entry/title', value='patched title')]
    with open_patched(nexus_file, ops) as f:
        assert f['entry/title'][()] == 'patched title'


def test_set_dataset_raises_if_parent_does_not_exist(nexus_file: Path) -> None:
    ops = [patch.SetDataset(path='/entry/missing/x', value=1)]
    with pytest.raises(KeyError, match='/entry/missing'):
        open_patched(nexus_file, ops)


def test_depends_on_resolves_in_patched_file(nexus_file: Path) -> None:
    ops = [
        patch.Move(
            source=f'{DETECTOR}/transformations/offset',
            target=f'{DETECTOR}/transformations/moved',
        ),
        patch.SetAttrs(path=f'{DETECTOR}/transformations/moved', attrs={'units': 'cm'}),
        patch.SetDataset(
            path=f'{DETECTOR}/depends_on', value=f'{DETECTOR}/transformations/moved'
        ),
    ]
    with open_patched(nexus_file, ops) as f:
        detector = f[DETECTOR][()]
    transformations = detector['depends_on'].transformations
    assert list(transformations) == [f'{DETECTOR}/transformations/moved']
    assert transformations[f'{DETECTOR}/transformations/moved'].value.unit == 'cm'


def test_soft_link_resolves_in_patched_file(nexus_file: Path) -> None:
    ops = [patch.SetDataset(path=f'{DETECTOR}/detector_number', value=[7, 8])]
    with open_patched(nexus_file, ops) as f:
        assert list(f['entry/number_link'][()].values) == [7, 8]


def test_load_patch_from_toml(tmp_path: Path) -> None:
    path = tmp_path / 'patch.toml'
    path.write_text(
        '''
[[op]]
kind = "set_attrs"
path = "/entry"
attrs = { vector = [0.0, 1.0], name = "x" }

[[op]]
kind = "move"
source = "/a"
target = "/b"
'''
    )
    assert patch.load_patch(path) == [
        patch.SetAttrs(path='/entry', attrs={'vector': [0.0, 1.0], 'name': 'x'}),
        patch.Move(source='/a', target='/b'),
    ]


def test_load_patch_raises_on_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / 'patch.toml'
    path.write_text('[[op]]\nkind = "rename"\npath = "/a"\n')
    with pytest.raises(ValueError, match="'rename'"):
        patch.load_patch(path)


def make_workflow(nexus_file: Path) -> GenericNeXusWorkflow:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = nexus_file
    wf[NeXusDetectorName] = 'detector'
    return wf


@pytest.mark.parametrize('preopen', [False, True])
def test_workflow_applies_patch(nexus_file: Path, preopen: bool) -> None:
    wf = make_workflow(nexus_file)
    wf[PreopenNeXusFile] = preopen
    wf[NeXusFilePatch[SampleRun]] = [
        patch.SetAttrs(path=f'{DETECTOR}/transformations/offset', attrs={'units': 'cm'})
    ]
    detector = wf.compute(EmptyDetector[SampleRun])
    sc.testing.assert_identical(
        detector.coords['position'][0], sc.vector([0.0, 0.0, 0.02], unit='m')
    )


def test_workflow_does_not_patch_by_default(nexus_file: Path) -> None:
    detector = make_workflow(nexus_file).compute(EmptyDetector[SampleRun])
    sc.testing.assert_identical(
        detector.coords['position'][0], sc.vector([0.0, 0.0, 2.0], unit='m')
    )
