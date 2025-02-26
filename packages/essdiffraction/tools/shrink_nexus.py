# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Make small DREAM nexus files for tests.

Note that this code modifies the file in-place.
Make sure to create a copy first.

This script keeps only 1 out of 16 detector pixels for each detector bank.
"""

import h5py as h5
import numpy as np

fname = 'DREAM_nexus_sorted-2023-12-07.nxs'

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
    # "sans_detector": {"strip": 32, "other": -1},
}

keys = DETECTOR_BANK_SIZES.keys()

with h5.File(fname, 'r+') as ds:
    for key in keys:
        print(key)  # noqa: T201
        base_path = f"entry/instrument/{key}"
        min_det_num = ds[base_path + '/detector_number'][()].min()
        # All first dimensions can be divided by 16
        keep = int(np.prod(list(DETECTOR_BANK_SIZES[key].values()))) // 16
        max_det_num = keep + min_det_num

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
        evids = ds[id_path][()]
        sel = evids < max_det_num

        ds[tmp] = ds[id_path]
        del ds[id_path]
        ds.create_dataset(id_path, data=evids[sel])
        ds[id_path].attrs.update(ds[tmp].attrs)
        del ds[tmp]

        eto_path = event_path + "/event_time_offset"
        etos = ds[eto_path][()]
        ds[tmp] = ds[eto_path]
        del ds[eto_path]
        ds.create_dataset(eto_path, data=etos[sel])
        ds[eto_path].attrs.update(ds[tmp].attrs)
        del ds[tmp]
