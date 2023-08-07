# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ess.v20.imaging.operations import groupby2D


def test_groupby2d_simple_case_neutron_specific():
    data = sc.array(
        dims=['wavelength', 'y', 'x'], values=np.arange(100.0).reshape(1, 10, 10)
    )
    wav = sc.scalar(value=1.0)
    x = sc.array(dims=['x'], values=np.arange(10))
    y = sc.array(dims=['y'], values=np.arange(10))
    source_position = sc.vector(value=[0, 0, -10])
    ds = sc.Dataset(
        data={'a': data},
        coords={'y': y, 'x': x, 'wavelength': wav, 'source_position': source_position},
    )
    grouped = groupby2D(ds, 5, 5)
    assert grouped['a'].shape == (1, 5, 5)
    grouped = groupby2D(ds, 1, 1)
    assert grouped['a'].shape == (1, 1, 1)
    assert 'source_position' in grouped['a'].meta


def _make_simple_dataset(u, v, w):
    data = sc.array(dims=['u', 'v', 'w'], values=np.ones((u, v, w)))
    u = sc.array(dims=['u'], values=np.arange(u))
    v = sc.array(dims=['v'], values=np.arange(v))
    w = sc.array(dims=['w'], values=np.arange(w))
    return sc.Dataset(
        data={'a': data},
        coords={
            'w': w,
            'v': v,
            'u': u,
        },
    )


def test_simple_case_any_naming():
    ds = _make_simple_dataset(u=2, v=10, w=10)
    grouped = groupby2D(ds, nx_target=5, ny_target=5, x='w', y='v', z='u')
    assert grouped['a'].shape == (2, 5, 5)
    projection = sc.array(dims=['v', 'w'], values=np.ones((5, 5))) * 4
    expected_data = sc.concat([projection, projection], dim='u')
    assert sc.all(
        sc.isclose(grouped['a'].data, expected_data, atol=1e-14 * sc.units.one)
    ).value


def test_groupby2d_different_output_size():
    ds = _make_simple_dataset(u=2, v=10, w=10)
    grouped = groupby2D(ds, nx_target=2, ny_target=5, x='w', y='v', z='u')
    assert grouped['a'].sizes['v'] == 5
    assert grouped['a'].sizes['w'] == 2
    assert grouped['a'].sizes['u'] == 2
