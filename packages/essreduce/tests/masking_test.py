# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

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


def _write_raw_pixel_mask(path: Path, field: str, values: np.ndarray) -> Path:
    with snx.File(path, 'w') as file:
        entry = file.create_class('entry', snx.NXentry)
        instrument = entry.create_class('instrument', snx.NXinstrument)
        detector = instrument.create_class('bank_0', snx.NXdetector)
        dataset = detector.create_field(field, values)
        dataset.dims[0].label = 'tube'
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


def test_read_bank_masks_names_mask_after_file_by_default(tmp_path: Path) -> None:
    mask = sc.array(dims=['tube'], values=[True, False])
    write_pixel_masks(tmp_path / 'a.nxs', {'bank_0': mask})
    masks = _read(tmp_path / 'a.nxs')
    assert sc.identical(masks['a.nxs']['bank_0'], mask)


def test_read_bank_masks_uses_name_stored_in_file(tmp_path: Path) -> None:
    mask = sc.array(dims=['tube'], values=[True, False])
    write_pixel_masks(tmp_path / 'a.nxs', {'bank_0': mask}, name='broken')
    assert list(_read(tmp_path / 'a.nxs')) == ['broken']


def test_read_bank_masks_keeps_dimensions_of_all_banks(tmp_path: Path) -> None:
    banks = {
        'bank_0': sc.array(dims=['layer', 'tube'], values=[[True, False]]),
        'bank_1': sc.array(dims=['straw'], values=[False, True, False]),
    }
    write_pixel_masks(tmp_path / 'a.nxs', banks)
    loaded = _read(tmp_path / 'a.nxs')['a.nxs']
    assert sc.identical(loaded, sc.DataGroup(banks))


def test_read_ignores_other_entries_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(path, {'bank_0': sc.array(dims=['tube'], values=[True])})
    with snx.File(path, 'r+') as file:
        file['entry'].create_field('valid_from', '2026-11-01T00:00:00Z')
    assert list(_read(path)['a.nxs']) == ['bank_0']


def test_read_skips_detector_without_pixel_mask(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(path, {'bank_0': sc.array(dims=['tube'], values=[True])})
    with snx.File(path, 'r+') as file:
        file['entry/instrument'].create_class('bank_1', snx.NXdetector)
    assert list(_read(path)['a.nxs']) == ['bank_0']


def test_read_raises_if_stored_names_are_equal(tmp_path: Path) -> None:
    mask = sc.array(dims=['tube'], values=[True])
    write_pixel_masks(tmp_path / 'a.nxs', {'bank_0': mask}, name='broken')
    write_pixel_masks(tmp_path / 'b.nxs', {'bank_0': mask}, name='broken')
    with pytest.raises(ValueError, match='broken'):
        _read(tmp_path / 'a.nxs', tmp_path / 'b.nxs')


@pytest.mark.parametrize('dtype', ['uint8', 'int32', 'int64'])
def test_read_masks_pixels_with_any_rejecting_bit(tmp_path: Path, dtype: str) -> None:
    values = np.array([0, 1, 2], dtype=dtype)
    path = _write_raw_pixel_mask(tmp_path / 'a.nxs', 'pixel_mask', values)
    loaded = _read(path)['a.nxs']['bank_0']
    assert loaded.dims == ('tube',)
    assert loaded.values.tolist() == [False, True, True]


def test_read_does_not_mask_pixels_with_tag_bits_only(tmp_path: Path) -> None:
    values = np.array([1 << 8, 1 << 31], dtype='uint32')
    path = _write_raw_pixel_mask(tmp_path / 'a.nxs', 'pixel_mask', values)
    assert _read(path)['a.nxs']['bank_0'].values.tolist() == [True, False]


def test_read_raises_if_pixel_mask_is_not_integer(tmp_path: Path) -> None:
    path = _write_raw_pixel_mask(tmp_path / 'a.nxs', 'pixel_mask', np.array([1.5]))
    with pytest.raises(ValueError, match='non-integer'):
        _read(path)


def test_read_raises_on_numbered_pixel_masks(tmp_path: Path) -> None:
    path = _write_raw_pixel_mask(tmp_path / 'a.nxs', 'pixel_mask', np.array([1]))
    with snx.File(path, 'r+') as file:
        file['entry/instrument/bank_0'].create_field('pixel_mask_2', np.array([1]))
    with pytest.raises(ValueError, match='pixel_mask_2'):
        _read(path)


def test_read_raises_on_several_entries(tmp_path: Path) -> None:
    path = tmp_path / 'a.nxs'
    write_pixel_masks(path, {'bank_0': sc.array(dims=['tube'], values=[True])})
    with snx.File(path, 'r+') as file:
        file.create_class('other', snx.NXentry)
    with pytest.raises(ValueError, match='several NXentry'):
        _read(path)


def test_read_raises_if_file_has_no_pixel_mask(tmp_path: Path) -> None:
    path = _write_raw_pixel_mask(tmp_path / 'a.nxs', 'detector_number', np.array([1]))
    with pytest.raises(ValueError, match='no detector with a pixel_mask'):
        _read(path)


def test_to_detector_masks_keeps_shape_of_ids() -> None:
    ids = sc.array(dims=['tube', 'pixel'], values=[[1, 2], [3, 4]], unit=None)
    masked = sc.array(dims=['detector_id'], values=[2, 3, 99], unit=None)
    masks = to_detector_masks(PixelMasks({'m': masked}), ids)
    expected = sc.array(dims=['tube', 'pixel'], values=[[False, True], [True, False]])
    assert sc.identical(masks['m'], expected)


@pytest.fixture
def bank_ids() -> sc.Variable:
    return sc.array(dims=['tube', 'pixel'], values=[[1, 2], [3, 4], [5, 6]], unit=None)


def _per_bank(**banks: sc.Variable) -> PixelMasks:
    return PixelMasks({'m': sc.DataGroup(banks)})


def test_to_detector_masks_selects_bank_and_allows_fewer_dims(
    bank_ids: sc.Variable,
) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    other = sc.array(dims=['tube'], values=[False, True, False])
    masks = to_detector_masks(_per_bank(a=mask, b=other), bank_ids, bank='a')
    assert sc.identical(masks['m'], mask)


def test_to_detector_masks_skips_file_without_this_bank(bank_ids: sc.Variable) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    assert to_detector_masks(_per_bank(b=mask), bank_ids, bank='a') == {}


def test_to_detector_masks_applies_ids_to_any_bank(bank_ids: sc.Variable) -> None:
    masked = sc.array(dims=['detector_id'], values=[1], unit=None)
    masks = to_detector_masks(PixelMasks({'m': masked}), bank_ids, bank='a')
    assert masks['m'].values.sum() == 1


def test_to_detector_masks_raises_without_bank_for_masks_per_bank(
    bank_ids: sc.Variable,
) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    with pytest.raises(ValueError, match='no bank is known'):
        to_detector_masks(_per_bank(a=mask), bank_ids)


def test_to_detector_masks_raises_on_unknown_bank(bank_ids: sc.Variable) -> None:
    mask = sc.array(dims=['tube'], values=[True, False, False])
    with pytest.raises(ValueError, match='typo'):
        to_detector_masks(
            _per_bank(a=mask, typo=mask), bank_ids, bank='a', known_banks={'a', 'b'}
        )


def test_to_detector_masks_raises_if_masked_ids_are_not_integer(
    bank_ids: sc.Variable,
) -> None:
    masked = sc.array(dims=['x'], values=[True])
    with pytest.raises(sc.DTypeError, match='integer detector IDs'):
        to_detector_masks(PixelMasks({'m': masked}), bank_ids)


def test_to_detector_masks_raises_if_bank_mask_is_not_boolean(
    bank_ids: sc.Variable,
) -> None:
    mask = sc.array(dims=['tube'], values=[0, 1, 2], unit=None)
    with pytest.raises(sc.DTypeError, match='boolean'):
        to_detector_masks(_per_bank(a=mask), bank_ids, bank='a')


@pytest.mark.parametrize(
    'mask',
    [
        sc.array(dims=['straw'], values=[True, False, False]),
        sc.array(dims=['tube'], values=[True, False]),
    ],
    ids=['unknown-dim', 'wrong-size'],
)
def test_to_detector_masks_raises_if_mask_does_not_fit_detector(
    bank_ids: sc.Variable, mask: sc.Variable
) -> None:
    with pytest.raises(sc.DimensionError):
        to_detector_masks(_per_bank(a=mask), bank_ids, bank='a')
