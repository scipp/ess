# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from ess.dream.instrument_view import InstrumentView


@pytest.fixture
def fake_instrument_data(modules=('bank1', 'bank2', 'bank3', 'bank4', 'bank5')):
    rng = np.random.default_rng()

    out = {}
    npix = 300
    ntof = 100
    locations = range(len(modules))
    for name, loc in zip(modules, locations, strict=True):
        position = rng.normal(loc=[0, 0, loc], scale=[0.2, 0.2, 0.05], size=[npix, 3])
        tof = sc.linspace('tof', 0, 1.0e5, ntof + 1, unit='us')
        values = rng.normal(loc=5.0e4, scale=2.0e4, size=[npix, ntof])
        vec = sc.vectors(dims=['pixel'], unit='m', values=position)
        out[name] = sc.DataArray(
            data=sc.array(dims=['pixel', 'tof'], values=values, unit='counts'),
            coords={
                'position': vec,
                'x': vec.fields.x,
                'y': vec.fields.y,
                'z': vec.fields.z,
                'tof': tof,
            },
        )
    return sc.DataGroup(out)


def test_instrument_view_all_modules(fake_instrument_data):
    view = InstrumentView(fake_instrument_data, dim='tof')
    assert hasattr(view, 'checkboxes')
    assert hasattr(view, 'fig')
    assert hasattr(view, 'slider')


def test_instrument_view_one_module(fake_instrument_data):
    view = InstrumentView(fake_instrument_data['bank1'], dim='tof')
    assert not hasattr(view, 'checkboxes')
    assert hasattr(view, 'fig')
    assert hasattr(view, 'slider')


def test_instrument_view_slider_not_last_dim_dataarray(fake_instrument_data):
    da = fake_instrument_data['bank1']
    da = da.transpose(('tof', *(set(da.dims) - {'tof'})))
    InstrumentView(da, dim='tof')


def test_instrument_view_slider_not_last_dim_datagroup(fake_instrument_data):
    da = fake_instrument_data
    # Add extra dim so that not all entries in the group have the same set of dimensions
    da['bank2'] = da['bank2'].broadcast(
        dims=[*da['bank2'].dims, 'extra_dimension'], shape=[*da['bank2'].shape, 1]
    )
    for k, v in da.items():
        da[k] = v.transpose(('tof', *(set(v.dims) - {'tof'})))
    InstrumentView(da, dim='tof')


def test_instrument_view_no_tof_slider(fake_instrument_data):
    view = InstrumentView(fake_instrument_data.sum('tof'))
    assert hasattr(view, 'checkboxes')
    assert hasattr(view, 'fig')
    assert not hasattr(view, 'slider')


def test_instrument_view_one_module_no_tof_slider(fake_instrument_data):
    view = InstrumentView(fake_instrument_data['bank3'].sum('tof'))
    assert not hasattr(view, 'checkboxes')
    assert hasattr(view, 'fig')
    assert not hasattr(view, 'slider')


def test_instrument_view_toggle_module(fake_instrument_data):
    view = InstrumentView(fake_instrument_data, dim='tof')
    for name in fake_instrument_data:
        key = view.artist_mapping[name]
        assert view.fig.artists[key].points.visible
        view.checkboxes[name].value = False
        assert not view.fig.artists[key].points.visible
