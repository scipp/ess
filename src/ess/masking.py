# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from pathlib import Path
from typing import Mapping, Optional, Union

import h5py
import scipp as sc


def save_detector_masks(
    filename: Union[str, Path],
    detectors: Mapping[str, sc.DataArray],
    *,
    entry_metadata: Optional[Mapping[str, Union[str, int]]] = None,
) -> None:
    """
    Save detector masks to an HDF5/NeXus file.

    The masks will be saved inside an NXentry/NXinstrument group of the file.
    An NXdetector group will be created for each detector.

    Parameters
    ----------
    filename:
        Name of the file to save to.
    detectors:
        Dictionary of data arrays whose masks should be saved.
    entry_metadata:
        Optional dictionary of metadata to save in the NXentry group. Typically this
        should contain 'experiment_identifier', 'start_time', and 'end_time'.
    """
    with h5py.File(filename, "w-") as f:
        entry = f.require_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        if entry_metadata:
            for key, value in entry_metadata.items():
                entry[key] = value
        instrument = entry.require_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        for name, det in detectors.items():
            _save_masks(instrument, detector_name=name, detector=det)


def _save_masks(instrument, detector_name, detector):
    group = instrument.create_group(detector_name)
    group.attrs["NX_class"] = "NXdetector"
    group.attrs["axes"] = ["."] * len(detector.dims)
    for mask_name, mask in detector.masks.items():
        ds = group.create_dataset(mask_name, data=mask.values)
        group.attrs[f"{mask_name}_indices"] = [
            detector.dims.index(dim) for dim in mask.dims
        ]
        for i, dim in enumerate(mask.dims):
            ds.dims[i].label = dim
