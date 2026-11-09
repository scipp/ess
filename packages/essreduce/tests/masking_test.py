# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import pytest
import scipp as sc

from ess.reduce.masking import (
    LoadedPixelMasks,
    PixelMaskFilenames,
    read_pixel_masks,
    to_pixel_masks,
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


def _read(*paths: Path) -> LoadedPixelMasks:
    return read_pixel_masks(PixelMaskFilenames(paths))


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


def test_to_pixel_masks_keeps_shape_of_ids() -> None:
    ids = sc.array(dims=['tube', 'pixel'], values=[[1, 2], [3, 4]], unit=None)
    masked = sc.array(dims=['detector_id'], values=[2, 3, 99], unit=None)
    masks = to_pixel_masks(LoadedPixelMasks({'m': masked}), ids)
    expected = sc.array(dims=['tube', 'pixel'], values=[[False, True], [True, False]])
    assert sc.identical(masks['m'], expected)
