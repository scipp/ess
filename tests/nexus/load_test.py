# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import h5py
import pytest
import scipp as sc
import scippnexus as snx
from ess import nexus


def create_event_data_ids_1234(group):
    group.create_field('event_id',
                       sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2]))
    group.create_field('event_time_offset',
                       sc.array(dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]))
    group.create_field('event_time_zero',
                       sc.array(dims=[''], unit='s', values=[1, 2, 3, 4]))
    group.create_field('event_index',
                       sc.array(dims=[''], unit='None', values=[0, 3, 3, 5]))


@pytest.fixture()
def nxroot(request):
    """
    Yield NXroot containing a single NXentry, NXinstrument, NXsample, NXsource,
    monitors, detectors, and logs.
    """
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.NXroot(f)
        # This is modelled after the basic recommended NeXus structure
        entry = root.create_class('entry0', snx.NXentry)
        sample = entry.create_class('sample0', snx.NXsample)
        # Event mode monitor
        mon0 = entry.create_class('monitor0', snx.NXmonitor)
        mon0['event_time_offset'] = sc.array(dims=[''],
                                             unit='s',
                                             values=[456, 7, 3, 345, 632, 23])
        mon0['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
        mon0['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 5])
        mon1 = entry.create_class('monitor1', snx.NXmonitor)
        mon1['values'] = sc.array(dims=['time'], unit='ms', values=[10, 12, 9])
        mon1.attrs['signal'] = 'values'
        mon1.attrs['axes'] = ['time']

        instrument = entry.create_class('instr', snx.NXinstrument)
        source = instrument.create_class('src', snx.NXsource)
        det0 = instrument.create_class('det0', snx.NXdetector)
        detector_numbers = sc.array(dims=[''], unit=None, values=[1, 2, 3, 4])
        det0.create_field('detector_number', detector_numbers)
        create_event_data_ids_1234(det0.create_class('events', snx.NXevent_data))
        det1 = instrument.create_class('det1', snx.NXdetector)
        detector_numbers = sc.array(dims=[''], unit=None, values=[5, 6, 7, 8])
        det1.create_field('detector_number', detector_numbers)
        create_event_data_ids_1234(det1.create_class('events', snx.NXevent_data))
        yield root


def test_load_monitors_descends_into_unique_entry(nxroot):
    monitors = nexus.load_monitors(nxroot)
    assert len(monitors) == 2


def test_load_monitors_loads_from_entry(nxroot):
    monitors = nexus.load_monitors(nxroot.entry)
    assert len(monitors) == 2


def test_load_detectors_descends_into_unique_entry(nxroot):
    detectors = nexus.load_detectors(nxroot)
    assert len(detectors) == 2


def test_load_detectors_loads_from_entry(nxroot):
    detectors = nexus.load_detectors(nxroot.entry)
    assert len(detectors) == 2
