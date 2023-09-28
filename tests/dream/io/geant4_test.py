# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import zipfile
from io import BytesIO
from typing import Optional, Set

import numpy as np
import pytest
import scipp as sc

from ess.dream import data, load_geant4_csv


@pytest.fixture(scope='module')
def file_path():
    return data.get_path('data_dream_with_sectors.csv.zip')


# Load file into memory only once
@pytest.fixture(scope='module')
def load_file(file_path):
    with zipfile.ZipFile(file_path, 'r') as archive:
        return archive.read(archive.namelist()[0])


@pytest.fixture(scope='function')
def file(load_file):
    return BytesIO(load_file)


def assert_index_coord(
    coord: sc.Variable, *, values: Optional[Set[int]] = None
) -> None:
    assert coord.ndim == 1
    assert coord.unit is None
    assert coord.dtype == 'int64'
    if values is not None:
        assert set(np.unique(coord.values)) == values


def test_load_geant4_csv_loads_expected_structure(file):
    loaded = load_geant4_csv(file)
    assert isinstance(loaded, sc.DataGroup)
    assert loaded.keys() == {'instrument'}

    instrument = loaded['instrument']
    assert isinstance(instrument, sc.DataGroup)
    assert instrument.keys() == {
        'mantle',
        'high_resolution',
        'endcap_forward',
        'endcap_backward',
    }


def test_load_geant4_csv_mantle_has_expected_coords(file):
    # Only testing ranges that will not change in the future
    mantle = load_geant4_csv(file)['instrument']['mantle']
    assert_index_coord(mantle.coords['module'])
    assert_index_coord(mantle.coords['segment'])
    assert_index_coord(mantle.coords['counter'])
    assert_index_coord(mantle.coords['wire'], values=set(range(1, 33)))
    assert_index_coord(mantle.coords['strip'], values=set(range(1, 257)))
    assert 'sector' not in mantle.coords

    assert 'sector' not in mantle.bins.coords
    assert 'tof' in mantle.bins.coords
    assert 'wavelength' in mantle.bins.coords
    assert 'position' in mantle.bins.coords


def test_load_geant4_csv_endcap_backward_has_expected_coords(file):
    endcap = load_geant4_csv(file)['instrument']['endcap_backward']
    assert_index_coord(endcap.coords['module'])
    assert_index_coord(endcap.coords['segment'])
    assert_index_coord(endcap.coords['counter'])
    assert_index_coord(endcap.coords['wire'], values=set(range(1, 17)))
    assert_index_coord(endcap.coords['strip'], values=set(range(1, 17)))
    assert 'sector' not in endcap.coords

    assert 'sector' not in endcap.bins.coords
    assert 'tof' in endcap.bins.coords
    assert 'wavelength' in endcap.bins.coords
    assert 'position' in endcap.bins.coords


def test_load_geant4_csv_endcap_forward_has_expected_coords(file):
    endcap = load_geant4_csv(file)['instrument']['endcap_forward']
    assert_index_coord(endcap.coords['module'])
    assert_index_coord(endcap.coords['segment'])
    assert_index_coord(endcap.coords['counter'])
    assert_index_coord(endcap.coords['wire'], values=set(range(1, 17)))
    assert_index_coord(endcap.coords['strip'], values=set(range(1, 17)))
    assert 'sector' not in endcap.coords

    assert 'sector' not in endcap.bins.coords
    assert 'tof' in endcap.bins.coords
    assert 'wavelength' in endcap.bins.coords
    assert 'position' in endcap.bins.coords


def test_load_geant4_csv_high_resolution_has_expected_coords(file):
    hr = load_geant4_csv(file)['instrument']['high_resolution']
    assert_index_coord(hr.coords['module'])
    assert_index_coord(hr.coords['segment'])
    assert_index_coord(hr.coords['counter'])
    assert_index_coord(hr.coords['wire'], values=set(range(1, 17)))
    assert_index_coord(hr.coords['strip'], values=set(range(1, 33)))
    assert_index_coord(hr.coords['sector'], values=set(range(1, 5)))

    assert 'tof' in hr.bins.coords
    assert 'wavelength' in hr.bins.coords
    assert 'position' in hr.bins.coords
