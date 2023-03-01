# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass

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
        # This is modeled after the basic recommended NeXus structure
        entry = root.create_class('entry0', snx.NXentry)
        entry['start_time'] = sc.scalar("yesterday")
        sample = entry.create_class('sample0', snx.NXsample)
        sample['temperature'] = sc.scalar(1.2, unit='K')
        # Event mode monitor
        mon0 = entry.create_class('monitor0', snx.NXmonitor)
        mon0['event_time_offset'] = sc.array(dims=[''],
                                             unit='s',
                                             values=[456, 7, 3, 345, 632, 23])
        mon0['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
        mon0['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, 5])
        mon1 = entry.create_class('monitor1', snx.NXmonitor)
        mon1['data'] = sc.array(dims=['time'], unit='ms', values=[10, 12, 9])
        mon1.attrs['axes'] = ['time']

        instrument = entry.create_class('instr', snx.NXinstrument)
        instrument['field'] = sc.scalar('abc')
        instrument.create_class('src', snx.NXsource)
        det0 = instrument.create_class('det0', snx.NXdetector)
        detector_numbers = sc.array(dims=[''], unit=None, values=[1, 2, 3, 4])
        det0.create_field('detector_number', detector_numbers)
        create_event_data_ids_1234(det0.create_class('events', snx.NXevent_data))
        det1 = instrument.create_class('det1', snx.NXdetector)
        detector_numbers = sc.array(dims=[''], unit=None, values=[5, 6, 7, 8])
        det1.create_field('detector_number', detector_numbers)
        create_event_data_ids_1234(det1.create_class('events', snx.NXevent_data))
        yield root


def test_instrument_from_nexus_loads_detectors_and_fields(nxroot):
    instrument = nexus.BasicInstrument.from_nexus(nxroot.entry)
    assert len(instrument.detectors) == 2
    assert len(instrument.fields) == 1


def test_entry_from_nexus_loads_instrument_monitors_sample_and_fields(nxroot):
    entry = nexus.BasicEntry.from_nexus(nxroot)
    assert len(entry.instrument.detectors) == 2
    assert len(entry.instrument.fields) == 1
    assert len(entry.monitors) == 2
    assert sc.identical(entry.sample['temperature'], sc.scalar(1.2, unit='K'))
    assert len(entry.fields) == 1
    assert entry.fields['start_time'] == "yesterday"


def filter_and_load(group):
    if group.name.split('/')[-1] == 'det1':
        return group[()]


FilteredDetectors = nexus.make_section("FilteredDetectors", snx.NXdetector,
                                       filter_and_load)


def test_filter_via_load_function(nxroot):
    assert len(nxroot.entry.instrument[snx.NXdetector]) == 2
    dets = FilteredDetectors.from_nexus(nxroot.entry.instrument)
    assert len(dets) == 1
    assert 'det1' in dets


@dataclass
class InstrumentWithoutDetectors(nexus.InstrumentMixin):
    fields: nexus.Fields


@dataclass
class EntryWithoutDetectorsOrSample(nexus.EntryMixin):
    fields: nexus.Fields
    instrument: InstrumentWithoutDetectors


def test_load_custom_entry(nxroot):
    data = EntryWithoutDetectorsOrSample.from_nexus(nxroot)
    assert len(data.fields) == 1
    assert len(data.instrument.fields) == 1
    assert not hasattr(data, 'sample')
    assert not hasattr(data.instrument, 'detectors')


DetectorByPulse = nexus.make_section("DetectorPyPulse", snx.NXdetector,
                                     lambda group: group.events[()])


@dataclass
class InstrumentWithDetectorsByPulse(nexus.InstrumentMixin):
    detectors: DetectorByPulse


def test_load_instrument_without_binning_to_pixels(nxroot):
    data = InstrumentWithDetectorsByPulse.from_nexus(nxroot.entry)
    assert set(data.detectors) == {'det0', 'det1'}
    assert 'event_time_zero' in data.detectors['det0'].coords


NamedLeaf = nexus.make_leaf("NamedLeaf", "det1")


def test_loaded_named_leaf(nxroot):
    data = NamedLeaf.from_nexus(nxroot.entry.instrument)
    assert sc.identical(data, nxroot.entry.instrument['det1'][()])


DetectorsWithKwarg = nexus.make_section("DetectorsWithKwarg", snx.NXdetector,
                                        lambda group, *, extra: extra)


@dataclass
class InstrumentKwarg(nexus.BasicInstrument):
    detectors: DetectorsWithKwarg


@dataclass
class EntryKwarg(nexus.BasicEntry):
    instrument: InstrumentKwarg


def test_from_nexus_passes_kwarg_through_tree(nxroot):
    with pytest.raises(TypeError):
        data = EntryKwarg.from_nexus(nxroot)
    data = EntryKwarg.from_nexus(nxroot, instrument={'detectors': {'extra': 1.2}})
    assert data.instrument.detectors['det0'] == 1.2
    assert data.instrument.detectors['det1'] == 1.2
    data = EntryKwarg.from_nexus(nxroot, instrument={'detectors': {'extra': 1.3}})
    assert data.instrument.detectors['det0'] == 1.3
    assert data.instrument.detectors['det1'] == 1.3
