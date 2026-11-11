# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
import scippnexus as snx

from ess.reduce.masking import (
    PixelMaskFilenames,
    PixelMasks,
    read_pixel_masks,
    to_detector_masks,
    write_pixel_masks,
)


def _write_xml(path: Path, detids: str) -> Path:
    path.write_text(
        '<?xml version="1.0"?>\n<detector-masking><group>'
        f'<detids>{detids}</detids></group></detector-masking>'
    )
    return path


def _write_ids(path: Path, values: list[int]) -> Path:
    sc.io.save_hdf5(
        sc.array(dims=['detector_id'], values=values, unit=None, dtype='int32'), path
    )
    return path


def _read(*paths: Path) -> PixelMasks:
    return read_pixel_masks(PixelMaskFilenames(paths))


def _write_raw_mask(path: Path, values: np.ndarray, label: str = 'tube') -> Path:
    with snx.File(path, 'w') as file:
        entry = file.create_class('entry', snx.NXentry)
        instrument = entry.create_class('instrument', snx.NXinstrument)
        detector = instrument.create_class('bank_0', snx.NXdetector)
        collection = detector.create_class('masks', snx.NXcollection)
        dataset = collection.create_field('m', values)
        dataset.dims[0].label = label
    return path


def test_read_no_files_gives_no_masks() -> None:
    assert read_pixel_masks(PixelMaskFilenames([])) == {}


def test_read_xml_expands_ranges(tmp_path: Path) -> None:
    masks = _read(_write_xml(tmp_path / 'a.xml', '1-3, 7'))
    assert masks['a.xml'].values.tolist() == [1, 2, 3, 7]


def test_read_hdf5_detector_ids(tmp_path: Path) -> None:
    masks = _read(_write_ids(tmp_path / 'a.h5', [4, 5]))
    assert masks['a.h5'].values.tolist() == [4, 5]


def test_read_combines_formats_and_names_masks_without_directory(
    tmp_path: Path,
) -> None:
    masks = _read(
        _write_xml(tmp_path / 'a.xml', '1'), _write_ids(tmp_path / 'b.hdf5', [2])
    )
    assert list(masks) == ['a.xml', 'b.hdf5']


def test_read_raises_if_two_files_have_same_name(tmp_path: Path) -> None:
    (tmp_path / 'x').mkdir()
    (tmp_path / 'y').mkdir()
    first = _write_xml(tmp_path / 'x' / 'a.xml', '1')
    second = _write_xml(tmp_path / 'y' / 'a.xml', '2')
    with pytest.raises(ValueError, match=r'a\.xml'):
        _read(first, second)


def test_read_raises_on_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='Unsupported suffix'):
        _read(tmp_path / 'a.txt')


def test_read_raises_if_hdf5_does_not_hold_integers(tmp_path: Path) -> None:
    path = tmp_path / 'a.h5'
    sc.io.save_hdf5(sc.array(dims=['x'], values=[1.5]), path)
    with pytest.raises(ValueError, match='integer detector IDs'):
        _read(path)


def test_read_bank_masks_names_masks_after_datasets(tmp_path: Path) -> None:
    broken = sc.array(dims=['tube'], values=[True, False])
    beamstop = sc.array(dims=['straw'], values=[False, True, True])
    write_pixel_masks(
        tmp_path / 'a.nxs',
        {'broken': {'bank_0': broken}, 'beamstop': {'bank_0': beamstop}},
        instrument='TEST',
    )
    masks = _read(tmp_path / 'a.nxs')
    assert sc.identical(masks['broken'], sc.DataGroup(bank_0=broken))
    assert sc.identical(masks['beamstop'], sc.DataGroup(bank_0=beamstop))


def test_read_bank_masks_keeps_dimensions_of_all_banks(tmp_path: Path) -> None:
    banks = {
        'bank_0': sc.array(dims=['layer', 'tube'], values=[[True, False]]),
        'bank_1': sc.array(dims=['straw'], values=[False, True, False]),
    }
    write_pixel_masks(tmp_path / 'a.nxs', {'m': banks}, instrument='TEST')
    assert sc.identical(_read(tmp_path / 'a.nxs')['m'], sc.DataGroup(banks))


def test_read_bank_masks_omits_banks_without_dataset_of_mask(tmp_path: Path) -> None:
    mask = sc.array(dims=['tube'], values=[True])
    write_pixel_masks(
        tmp_path / 'a.nxs',
        {'broken': {'bank_0': mask}, 'beamstop': {'bank_1': mask}},
        instrument='TEST',
    )
    masks = _read(tmp_path / 'a.nxs')
    assert list(masks['broken']) == ['bank_0']
    assert list(masks['beamstop']) == ['bank_1']


def test_read_ignores_other_entries_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(
        path,
        {'m': {'bank_0': sc.array(dims=['tube'], values=[True])}},
        instrument='TEST',
    )
    with snx.File(path, 'r+') as file:
        file['entry'].create_field('valid_from', '2026-11-01T00:00:00Z')
        file['entry/instrument/bank_0'].create_field('pixel_mask', np.array([1]))
    assert list(_read(path)) == ['m']


def test_read_skips_detector_without_masks(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(
        path,
        {'m': {'bank_0': sc.array(dims=['tube'], values=[True])}},
        instrument='TEST',
    )
    with snx.File(path, 'r+') as file:
        file['entry/instrument'].create_class('bank_1', snx.NXdetector)
    assert list(_read(path)['m']) == ['bank_0']


def test_read_combines_masks_of_several_files(tmp_path: Path) -> None:
    mask = sc.array(dims=['tube'], values=[True])
    write_pixel_masks(
        tmp_path / 'a.nxs', {'broken': {'bank_0': mask}}, instrument='TEST'
    )
    write_pixel_masks(
        tmp_path / 'b.nxs', {'beamstop': {'bank_0': mask}}, instrument='TEST'
    )
    masks = _read(tmp_path / 'a.nxs', tmp_path / 'b.nxs')
    assert list(masks) == ['broken', 'beamstop']


def test_read_raises_if_files_have_mask_with_same_name(tmp_path: Path) -> None:
    mask = sc.array(dims=['tube'], values=[True])
    write_pixel_masks(
        tmp_path / 'a.nxs', {'broken': {'bank_0': mask}}, instrument='TEST'
    )
    write_pixel_masks(
        tmp_path / 'b.nxs', {'broken': {'bank_1': mask}}, instrument='TEST'
    )
    with pytest.raises(ValueError, match='broken'):
        _read(tmp_path / 'a.nxs', tmp_path / 'b.nxs')


def test_write_stores_masks_as_uint8(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(
        path,
        {'m': {'bank_0': sc.array(dims=['tube'], values=[True])}},
        instrument='TEST',
    )
    with h5py.File(path) as file:
        assert file['entry/instrument/bank_0/masks/m'].dtype == 'uint8'


def test_write_records_instrument_creator_and_time(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    mask = sc.array(dims=['tube'], values=[True])
    before = datetime.now(UTC).replace(microsecond=0)
    write_pixel_masks(path, {'m': {'bank_0': mask}}, instrument='LOKI')
    with h5py.File(path) as file:
        assert file['entry/instrument/name'][()] == b'LOKI'
        assert file.attrs['creator'].startswith('essreduce ')
        assert before <= datetime.fromisoformat(file.attrs['file_time'])


@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int32', 'uint32', 'int64'])
def test_read_masks_pixels_with_nonzero_value(tmp_path: Path, dtype: str) -> None:
    values = np.array([0, 1, 1], dtype=dtype)
    if dtype != 'bool':
        values[2] = 2
    loaded = _read(_write_raw_mask(tmp_path / 'a.nxs', values))['m']['bank_0']
    assert loaded.dims == ('tube',)
    assert loaded.values.tolist() == [False, True, True]


def test_read_raises_if_mask_is_not_integer(tmp_path: Path) -> None:
    path = _write_raw_mask(tmp_path / 'a.nxs', np.array([1.5]))
    with pytest.raises(ValueError, match='not integer'):
        _read(path)


def test_read_raises_if_mask_has_no_dimension_labels(tmp_path: Path) -> None:
    path = _write_raw_mask(tmp_path / 'a.nxs', np.array([0, 1, 0]), label='')
    with pytest.raises(ValueError, match='dimension labels'):
        _read(path)


def test_read_raises_on_group_in_masks(tmp_path: Path) -> None:
    path = _write_raw_mask(tmp_path / 'a.nxs', np.array([1]))
    with snx.File(path, 'r+') as file:
        file['entry/instrument/bank_0/masks'].create_class('sub', snx.NXcollection)
    with pytest.raises(ValueError, match='group'):
        _read(path)


def test_read_raises_on_several_entries(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(
        path,
        {'m': {'bank_0': sc.array(dims=['tube'], values=[True])}},
        instrument='TEST',
    )
    with snx.File(path, 'r+') as file:
        file.create_class('other', snx.NXentry)
    with pytest.raises(ValueError, match='several NXentry'):
        _read(path)


def test_read_raises_if_file_has_no_masks(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    with snx.File(path, 'w') as file:
        entry = file.create_class('entry', snx.NXentry)
        instrument = entry.create_class('instrument', snx.NXinstrument)
        instrument.create_class('bank_0', snx.NXdetector)
    with pytest.raises(ValueError, match='no detector with masks'):
        _read(path)


def _detector(ids: sc.Variable, id_coord: str = 'detector_number') -> sc.DataArray:
    return sc.DataArray(sc.zeros(sizes=ids.sizes), coords={id_coord: ids})


def test_to_detector_masks_keeps_shape_of_ids() -> None:
    ids = sc.array(dims=['tube', 'pixel'], values=[[1, 2], [3, 4]], unit=None)
    masked = sc.array(dims=['detector_id'], values=[2, 3, 99], unit=None)
    masks = to_detector_masks(PixelMasks({'m': masked}), _detector(ids))
    expected = sc.array(dims=['tube', 'pixel'], values=[[False, True], [True, False]])
    assert sc.identical(masks['m'], expected)


def test_to_detector_masks_reads_ids_from_given_coord() -> None:
    ids = sc.array(dims=['spectrum'], values=[1, 2, 3], unit=None)
    masked = sc.array(dims=['detector_id'], values=[2], unit=None)
    masks = to_detector_masks(
        PixelMasks({'m': masked}), _detector(ids, 'detector_id'), id_coord='detector_id'
    )
    assert sc.identical(
        masks['m'], sc.array(dims=['spectrum'], values=[False, True, False])
    )


def test_to_detector_masks_does_not_need_ids_for_masks_per_bank() -> None:
    detector = sc.DataArray(sc.zeros(sizes={'tube': 3, 'pixel': 2}))
    mask = sc.array(dims=['tube'], values=[True, False, False])
    masks = to_detector_masks(_per_bank(a=mask), detector, bank='a')
    assert sc.identical(masks['m'], mask)


@pytest.fixture
def bank() -> sc.DataArray:
    ids = sc.array(dims=['tube', 'pixel'], values=[[1, 2], [3, 4], [5, 6]], unit=None)
    return _detector(ids)


def _per_bank(**banks: sc.Variable) -> PixelMasks:
    return PixelMasks({'m': sc.DataGroup(banks)})


def test_to_detector_masks_selects_bank_and_allows_fewer_dims(
    bank: sc.DataArray,
) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    other = sc.array(dims=['tube'], values=[False, True, False])
    masks = to_detector_masks(_per_bank(a=mask, b=other), bank, bank='a')
    assert sc.identical(masks['m'], mask)


def test_to_detector_masks_skips_file_without_this_bank(bank: sc.DataArray) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    assert to_detector_masks(_per_bank(b=mask), bank, bank='a') == {}


def test_to_detector_masks_applies_ids_to_any_bank(bank: sc.DataArray) -> None:
    masked = sc.array(dims=['detector_id'], values=[1], unit=None)
    masks = to_detector_masks(PixelMasks({'m': masked}), bank, bank='a')
    assert masks['m'].values.sum() == 1


def test_to_detector_masks_raises_without_bank_for_masks_per_bank(
    bank: sc.DataArray,
) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    with pytest.raises(ValueError, match='no bank is known'):
        to_detector_masks(_per_bank(a=mask), bank)


def test_to_detector_masks_raises_on_unknown_bank(bank: sc.DataArray) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    with pytest.raises(ValueError, match='typo'):
        to_detector_masks(
            _per_bank(a=mask, typo=mask), bank, bank='a', known_banks={'a', 'b'}
        )


def test_to_detector_masks_raises_if_masked_ids_are_not_integer(
    bank: sc.DataArray,
) -> None:
    masked = sc.array(dims=['x'], values=[True])
    with pytest.raises(sc.DTypeError, match='integer detector IDs'):
        to_detector_masks(PixelMasks({'m': masked}), bank)


def test_to_detector_masks_raises_if_bank_mask_is_not_boolean(
    bank: sc.DataArray,
) -> None:
    mask = sc.array(dims=['tube'], values=[0, 1, 2], unit=None)
    with pytest.raises(sc.DTypeError, match='boolean'):
        to_detector_masks(_per_bank(a=mask), bank, bank='a')


@pytest.mark.parametrize(
    'mask',
    [
        sc.array(dims=['straw'], values=[True, False, False]),
        sc.array(dims=['tube'], values=[True, False]),
        sc.array(dims=['pixel', 'tube'], values=[[True, False]] * 3),
        sc.array(dims=['tube', 'straw'], values=[[True], [False], [False]]),
    ],
    ids=['unknown-dim', 'wrong-size', 'sizes-of-other-dim', 'extra-dim'],
)
def test_to_detector_masks_raises_if_mask_does_not_fit_detector(
    bank: sc.DataArray, mask: sc.Variable
) -> None:
    with pytest.raises(sc.DimensionError):
        to_detector_masks(_per_bank(a=mask), bank, bank='a')


@pytest.mark.parametrize(
    ('mask_dims', 'expected_dims'),
    [
        (('layer', 'tube', 'straw'), ('layer', 'tube', 'straw')),
        (('straw', 'tube', 'layer'), ('layer', 'tube', 'straw')),
        (('tube', 'straw', 'layer'), ('layer', 'tube', 'straw')),
        (('tube',), ('tube',)),
        (('layer', 'straw'), ('layer', 'straw')),
        (('straw', 'layer'), ('layer', 'straw')),
        (('straw', 'tube'), ('tube', 'straw')),
    ],
    ids=[
        'full',
        'full-reversed',
        'full-reordered',
        'one-dim',
        'skips-dim',
        'skips-dim-transposed',
        'subset-transposed',
    ],
)
def test_to_detector_masks_follows_dimension_order_of_detector(
    mask_dims: tuple[str, ...], expected_dims: tuple[str, ...]
) -> None:
    sizes = {'layer': 2, 'tube': 3, 'straw': 4}
    detector = sc.DataArray(sc.zeros(sizes=sizes))
    shape = [sizes[dim] for dim in mask_dims]
    values = np.arange(np.prod(shape)).reshape(shape) % 3 == 0
    mask = sc.array(dims=list(mask_dims), values=values)
    result = to_detector_masks(_per_bank(a=mask), detector, bank='a')['m']
    assert result.dims == expected_dims
    assert sc.identical(result, mask.transpose(list(expected_dims)))


def test_to_detector_masks_accepts_scalar_for_whole_bank(
    bank: sc.DataArray,
) -> None:
    result = to_detector_masks(_per_bank(a=sc.scalar(True)), bank, bank='a')['m']
    assert sc.identical(result, sc.scalar(True))
