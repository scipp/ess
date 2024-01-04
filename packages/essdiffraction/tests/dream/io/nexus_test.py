# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

from ess import dream


@pytest.fixture
def filename():
    return dream.data.get_path('DREAM_nexus_sorted-2023-12-07.nxs')


@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_bunker")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_cave")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/polarizer/rate")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/sans_detector")
def test_load_nexus_loads_file(filename):
    dg = dream.load_nexus(filename)
    assert 'instrument' in dg
    instr = dg['instrument']
    for name in (
        'mantle',
        'endcap_backward',
        'endcap_forward',
        'high_resolution',
        'sans',
    ):
        assert f'{name}_detector' in instr
        det = instr[f'{name}_detector']
        assert 'pixel_shape' not in det


def test_load_nexus_fails_if_entry_not_found(filename):
    with pytest.raises(KeyError):
        dream.load_nexus(filename, entry='foo')


@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_bunker")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_cave")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/polarizer/rate")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/sans_detector")
def test_load_nexus_folds_detectors_by_default(filename):
    dg = dream.load_nexus(filename)
    instr = dg['instrument']
    # sans_detector is not populated in the current files
    for name in ('mantle', 'endcap_backward', 'endcap_forward', 'high_resolution'):
        det = instr[f'{name}_detector']
        # There may be other dims, but some are irregular and this may be subject to
        # change
        assert 'strip' in det.dims


@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_bunker")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_cave")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/polarizer/rate")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/sans_detector")
def test_load_nexus_with_disabled_fold(filename):
    dg = dream.load_nexus(filename, fold_detectors=False)
    instr = dg['instrument']
    for name in ('mantle', 'endcap_backward', 'endcap_forward', 'high_resolution'):
        det = instr[f'{name}_detector']
        assert det.dims == ('detector_number',)


@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_bunker")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/monitor_cave")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/polarizer/rate")
@pytest.mark.filterwarnings("ignore:Failed to load /entry/instrument/sans_detector")
def test_load_nexus_with_pixel_shape(filename):
    dg = dream.load_nexus(filename, load_pixel_shape=True)
    assert 'instrument' in dg
    instr = dg['instrument']
    # sans_detector is not populated in the current files
    for name in ('mantle', 'endcap_backward', 'endcap_forward', 'high_resolution'):
        assert f'{name}_detector' in instr
        det = instr[f'{name}_detector']
        assert 'pixel_shape' in det
