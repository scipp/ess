# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scitiff.io import load_scitiff

from ess import imaging as img
from ess.imaging.data import get_siemens_star_path


def test_blockify() -> None:
    da = load_scitiff(get_siemens_star_path())["image"]
    blocks = img.tools.blockify(da, {'x': 4, 'y': 4})
    assert len(blocks.dims) == len(da.dims) + 2
    assert {da.sizes['x'] // 4, da.sizes['y'] // 4, 4}.issubset(blocks.sizes.values())


def test_resample() -> None:
    da = load_scitiff(get_siemens_star_path())["image"]
    resampled = img.tools.resample(da, sizes={'x': 128, 'y': 128})
    assert resampled.sizes['x'] == 128
    assert resampled.sizes['y'] == 128
    assert sc.identical(resampled.sum(), da.sum())


def test_resample_mean() -> None:
    da = load_scitiff(get_siemens_star_path())["image"]
    resampled = img.tools.resample(da, sizes={'x': 128, 'y': 128}, method='mean')
    assert resampled.sizes['x'] == 128
    assert resampled.sizes['y'] == 128
    assert resampled.sum().value < da.sum().value


def test_resample_callable() -> None:
    da = load_scitiff(get_siemens_star_path())["image"]
    resampled = img.tools.resample(da, sizes={'x': 256, 'y': 256}, method=sc.max)
    assert resampled.sizes['x'] == 256
    assert resampled.sizes['y'] == 256


def test_resample_bad_size_requested_raises():
    da = load_scitiff(get_siemens_star_path())["image"]
    with pytest.raises(ValueError, match="Size of dimension 'x' .* is not divisible"):
        img.tools.resample(da, sizes={'x': 127, 'y': 127})


def test_laplace_2d() -> None:
    da = load_scitiff(get_siemens_star_path())["image"]
    resampled = img.tools.resample(da, sizes={'x': 2, 'y': 2})
    laplacian = img.tools.laplace_2d(resampled, dims=('x', 'y'))
    assert laplacian.sizes == resampled.sizes


def test_sharpness() -> None:
    da = load_scitiff(get_siemens_star_path())["image"]
    sharp = img.tools.sharpness(da, dims=('x', 'y'))
    assert (sharp['t', 0] < sharp['t', 1]).value
    assert (sharp['t', 0] > sharp['t', 2]).value
    assert (sharp['t', 1] > sharp['t', 2]).value
