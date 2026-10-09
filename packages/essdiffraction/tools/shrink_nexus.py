# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Make small DREAM nexus files for tests.

Note that this code modifies the file in-place.
Make sure to create a copy first.

This script keeps only 1 out of 16 detector pixels for each detector bank
and only the first 5 pulses.
"""

import h5py as h5
import numpy as np

fname = 'coda_dream_999999_00024365.hdf'

DETECTOR_BANK_SIZES = {
    "endcap_backward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 11,
        "segment": 28,
        "counter": 2,
    },
    "endcap_forward_detector": {
        "strip": 16,
        "wire": 16,
        "module": 5,
        "segment": 28,
        "counter": 2,
    },
    "mantle_detector": {
        "wire": 32,
        "module": 5,
        "segment": 6,
        "strip": 256,
        "counter": 2,
    },
    "high_resolution_detector": {"strip": 32, "other": -1},
    "sans_detector": {"strip": 32, "other": -1},
}

keys = DETECTOR_BANK_SIZES.keys()

with h5.File(fname, 'r+') as ds:
    for key in keys:
        print(key)  # noqa: T201
        base_path = f"entry/instrument/{key}"
        # min_det_num = ds[base_path + '/detector_number'][()].min()
        # All first dimensions can be divided by 16
        keep = int(np.prod(list(DETECTOR_BANK_SIZES[key].values()))) // 16
        # max_det_num = keep + min_det_num

        del ds[base_path + '/pixel_shape']
        tmp = base_path + "/tmp"  # noqa: S108
        for field in (
            'detector_number',
            'x_pixel_offset',
            'y_pixel_offset',
            'z_pixel_offset',
        ):
            here = base_path + f"/{field}"
            old = ds[here][()]
            ds[tmp] = ds[here]
            del ds[here]  # delete old, differently sized dataset
            ds.create_dataset(here, data=old[:keep])
            ds[here].attrs.update(ds[tmp].attrs)
            del ds[tmp]

        event_path = base_path + f"/{key.replace('detector', 'event_data')}"
        tmp = event_path + "/temp_path"

        # Select only events in the first set of detector pixels
        id_path = event_path + "/event_id"
        index_path = event_path + "/event_index"
        event_index = ds[index_path][()]
        zero_path = event_path + "/event_time_zero"
        pulse_count = min(5, ds[zero_path].shape[0])
        if pulse_count == 0:
            event_stop = 0
        elif pulse_count < event_index.size:
            event_stop = int(event_index[pulse_count])
        else:
            event_stop = ds[id_path].shape[0]
        event_index = event_index[:pulse_count]
        evids = ds[id_path][:event_stop]
        min_det_num = ds[base_path + '/detector_number'][()].min()
        max_det_num = ds[base_path + '/detector_number'][()].max()
        sel = (evids >= min_det_num) & (evids <= max_det_num)

        ds[tmp] = ds[id_path]
        del ds[id_path]
        ds.create_dataset(id_path, data=evids[sel])
        ds[id_path].attrs.update(ds[tmp].attrs)
        del ds[tmp]

        eto_path = event_path + "/event_time_offset"
        etos = ds[eto_path][:event_stop]
        ds[tmp] = ds[eto_path]
        del ds[eto_path]
        ds.create_dataset(eto_path, data=etos[sel])
        ds[eto_path].attrs.update(ds[tmp].attrs)
        del ds[tmp]

        # Need to re-create the event_index dataset with only the selected events
        # The event_index is an array that indicates the start of events for each
        # event_time_zero (meaning it encodes how many events are in each pulse)
        selected_before = np.empty(sel.size + 1, dtype=np.int64)
        selected_before[0] = 0
        np.cumsum(sel, dtype=np.int64, out=selected_before[1:])
        new_event_index = selected_before[event_index].astype(event_index.dtype)

        ds[tmp] = ds[index_path]
        del ds[index_path]
        ds.create_dataset(index_path, data=new_event_index)
        ds[index_path].attrs.update(ds[tmp].attrs)
        del ds[tmp]

        event_time_zero = ds[zero_path][:pulse_count]
        ds[tmp] = ds[zero_path]
        del ds[zero_path]
        ds.create_dataset(zero_path, data=event_time_zero)
        ds[zero_path].attrs.update(ds[tmp].attrs)
        del ds[tmp]
