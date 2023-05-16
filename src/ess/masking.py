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

    Examples
    --------

    Consider a data group with two detectors, one with a mask and one without, as well
    as some metadata:

      >>> import scipp as sc
      >>>
      >>> dg = sc.DataGroup()
      >>> dg['mantle_detector'] = sc.DataArray(
      ...     data=sc.ones(dims=['tube', 'pixel'], shape=[3, 100]),
      ...     masks={'bad_tube': sc.array(dims=['tube'], values=[False, True, False])},
      ... )
      >>> dg['endcap_detector'] = sc.DataArray(
      ...     data=sc.ones(dims=['x', 'y'], shape=[2, 2]), masks={}
      ... )
      >>> metadata = {'experiment_identifier': 12345}

    Save the masks to a file:

      >>> from ess.masking import save_detector_masks
      >>>
      >>> save_detector_masks('masks.h5', dg, entry_metadata=metadata)

    Use ScippNexus to load the resulting file:

      >>> import scippnexus.v2 as snx
      >>>
      >>> with snx.File('masks.h5') as f:
      >>>     masks = f['entry'][()]
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
