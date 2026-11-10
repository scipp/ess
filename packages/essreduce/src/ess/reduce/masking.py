# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Pixel masks stored in files.

Reading is independent of runs and detector banks, so a workflow reads its mask files
once. Converting the file content to masks for a detector is a separate step.
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import NewType

import numpy as np
import scipp as sc
import scippnexus as snx

PixelMaskFilenames = NewType('PixelMaskFilenames', Sequence[str | Path])
"""Files with pixel masks to apply. Use an empty sequence to apply none."""

PixelMasks = NewType('PixelMasks', dict[str, sc.Variable | sc.DataGroup])
"""Pixel masks of the whole instrument, by mask name.

Each value is either a variable of masked detector IDs, or a data group of boolean masks
by detector bank. :py:func:`read_pixel_masks` provides them from files, and they can
also be set directly. :py:func:`to_detector_masks` selects the masks for one detector.
"""

_HDF5_SUFFIXES = ('.h5', '.hdf', '.hdf5', '.nxs')


def read_pixel_masks(filenames: PixelMaskFilenames) -> PixelMasks:
    """Read pixel masks from files.

    The following formats are supported, selected by file suffix and content:

    - ``.xml``: Masked detector IDs in the ISIS format,
      see :py:func:`read_xml_masked_detector_ids`.
    - ``.h5``, ``.hdf``, ``.hdf5``, ``.nxs``: Either of

      - A NeXus file with named masks by detector bank,
        see :py:func:`write_pixel_masks`:

        .. code-block:: text

            /entry                         (NXentry)
                instrument                 (NXinstrument)
                    loki_detector_0        (NXdetector)
                        masks              (NXcollection)
                            broken_tubes   integer array
                            beamstop       integer array

        Bank names are the names of the NeXus detector groups, and mask names are the
        names of the datasets. A mask with the same name in several banks is one mask.
        A bank without a dataset of that name has no pixels masked by it. A pixel is
        masked if its value is not zero. The dimensions of a dataset are given by HDF5
        dimension labels. They are named after the logical dimensions of the bank, and
        may be a subset of them, in any order. Other entries of the file, such as a
        validity date, are ignored.
      - A file written by :py:func:`scipp.io.save_hdf5` with a variable of masked
        detector IDs. The mask is named after the file, without the directory, since
        directories differ between hosts. The same applies to ``.xml`` files.

    Parameters
    ----------
    filenames:
        Paths to the mask files.

    Raises
    ------
    ValueError
        If a file has an unsupported suffix or content, or two files give the same mask
        name. One mask would replace the other otherwise.
    """
    masks = {}
    for filename in map(Path, filenames):
        for name, content in _read_mask_file(filename).items():
            if name in masks:
                raise ValueError(f"Several mask files give the mask name '{name}'.")
            masks[name] = content
    return PixelMasks(masks)


def to_detector_masks(
    pixel_masks: PixelMasks,
    ids: sc.Variable,
    bank: str | None = None,
    known_banks: Collection[str] | None = None,
) -> dict[str, sc.Variable]:
    """Create boolean masks for the pixels of a detector bank.

    The masks use the logical dimensions of the bank, which are the dimensions of
    ``ids``:

    - Masks given per bank are selected by ``bank``. They keep their dimensions, which
      may be a subset of those of ``ids``, and are transposed to the order of ``ids``.
    - Masks given as a list of detector IDs, such as ISIS XML files, apply to any bank,
      since IDs identify a pixel. They are converted to a mask with the dimensions of
      ``ids`` by looking up each pixel's ID in the list.

    Parameters
    ----------
    pixel_masks:
        Pixel masks of the instrument.
    ids:
        Detector IDs of the pixels.
    bank:
        Name of the detector bank. Without it, masks given per bank raise an error.
    known_banks:
        Names of all banks of the instrument. If given, a file with masks for any other
        bank raises an error. A misspelled name would otherwise mask nothing.
    """
    masks = {}
    for name, content in pixel_masks.items():
        if isinstance(content, sc.Variable):
            masked_ids = _validate_masked_ids(content, name)
            masks[name] = sc.array(
                dims=ids.dims, values=np.isin(ids.values, masked_ids.values)
            )
            continue
        if bank is None:
            raise ValueError(f"Mask '{name}' is given per bank, but no bank is known.")
        if known_banks is not None and (unknown := set(content) - set(known_banks)):
            raise ValueError(f"Mask '{name}' has unknown banks: {sorted(unknown)}.")
        if bank in content:
            masks[name] = _align_bank_mask(content[bank], ids, name)
    return masks


def write_pixel_masks(
    filename: str | Path, masks: Mapping[str, Mapping[str, sc.Variable]]
) -> None:
    """Write named boolean masks by detector bank to a NeXus file.

    See :py:func:`read_pixel_masks` for the layout of the file. Masks are stored as
    unsigned 8-bit integers, since HDF5 has no boolean type.

    Parameters
    ----------
    filename:
        Path of the file to write.
    masks:
        Boolean masks by mask name and detector bank. Dimensions are named after the
        logical dimensions of the bank, and may be a subset of them, in any order.
    """
    with snx.File(filename, 'w') as file:
        entry = file.create_class('entry', snx.NXentry)
        instrument = entry.create_class('instrument', snx.NXinstrument)
        collections = {}
        for name, banks in masks.items():
            for bank, mask in banks.items():
                if bank not in collections:
                    detector = instrument.create_class(bank, snx.NXdetector)
                    collections[bank] = detector.create_class('masks', snx.NXcollection)
                dataset = collections[bank].create_field(
                    name, mask.values.astype('uint8')
                )
                for axis, dim in zip(dataset.dims, mask.dims, strict=True):
                    axis.label = dim


def read_xml_masked_detector_ids(filename: str | Path) -> sc.Variable:
    """Read a pixel mask from an ISIS XML file.

    The format is as follows, where the detids are inclusive ranges of detector IDs:

    .. code-block:: xml

        <?xml version="1.0"?>
        <detector-masking>
            <group>
                <detids>1400203-1400218,1401199,1402190-1402223</detids>
            </group>
        </detector-masking>

    Parameters
    ----------
    filename:
        Path to the XML file.
    """
    import xml.etree.ElementTree as ET  # nosec

    tree = ET.parse(filename)  # noqa: S314
    root = tree.getroot()

    masked_detids = []
    for group in root.findall('group'):
        for detids in group.findall('detids'):
            for detid in detids.text.split(','):
                detid = detid.strip()
                if '-' in detid:
                    start, stop = detid.split('-')
                    masked_detids += list(range(int(start), int(stop) + 1))
                else:
                    masked_detids.append(int(detid))

    return sc.array(
        dims=['detector_id'], values=masked_detids, unit=None, dtype='int32'
    )


def _read_mask_file(filename: Path) -> dict[str, sc.Variable | sc.DataGroup]:
    if filename.suffix == '.xml':
        return {filename.name: read_xml_masked_detector_ids(filename)}
    if filename.suffix in _HDF5_SUFFIXES:
        return _read_hdf5(filename)
    raise ValueError(f"Unsupported suffix of mask file '{filename}'.")


def _read_hdf5(filename: Path) -> dict[str, sc.Variable | sc.DataGroup]:
    with snx.File(filename) as file:
        # Reading only one of several entries would leave pixels unmasked.
        if len(file[snx.NXentry]) > 1:
            raise ValueError(f"Mask file '{filename}' has several NXentry groups.")
        if 'entry' in file:
            return _read_nexus_masks(file['entry'], filename)
    ids = sc.io.load_hdf5(filename)
    if isinstance(ids, sc.Variable) and ids.dtype in ('int32', 'int64'):
        return {filename.name: ids}
    raise ValueError(
        f"Mask file '{filename}' holds neither a NeXus entry nor integer detector IDs."
    )


def _read_nexus_masks(entry: snx.Group, filename: Path) -> dict[str, sc.DataGroup]:
    masks = {}
    for bank, detector in entry['instrument'][snx.NXdetector].items():
        if 'masks' not in detector:
            continue
        for name, field in detector['masks'].items():
            masks.setdefault(name, {})[bank] = _read_bank_mask(field, filename)
    if not masks:
        raise ValueError(f"Mask file '{filename}' has no detector with masks.")
    return {name: sc.DataGroup(banks) for name, banks in masks.items()}


def _read_bank_mask(field: snx.Field | snx.Group, filename: Path) -> sc.Variable:
    if not isinstance(field, snx.Field):
        raise ValueError(f"Mask file '{filename}' has a group '{field.name}' in masks.")
    dataset = field.dataset
    values = dataset[()]
    # h5py reads its boolean enumeration as bool.
    if not (values.dtype == bool or np.issubdtype(values.dtype, np.integer)):
        raise ValueError(f"Mask '{dataset.name}' in '{filename}' is not integer.")
    # scippnexus does not read HDF5 dimension labels of fields in an NXcollection.
    dims = [axis.label for axis in dataset.dims]
    if not all(dims):
        raise ValueError(
            f"Mask '{dataset.name}' in '{filename}' lacks HDF5 dimension labels."
        )
    return sc.array(dims=dims, values=values != 0)


def _validate_masked_ids(masked_ids: sc.Variable, name: str) -> sc.Variable:
    if masked_ids.dtype not in ('int32', 'int64'):
        raise sc.DTypeError(
            f"Mask '{name}' must hold integer detector IDs, not {masked_ids.dtype}."
        )
    return masked_ids


def _align_bank_mask(mask: sc.Variable, ids: sc.Variable, name: str) -> sc.Variable:
    if mask.dtype != 'bool':
        raise sc.DTypeError(f"Mask '{name}' must be boolean, not {mask.dtype}.")
    # Masks with a dimension that the data lacks would be ignored without an error.
    if not mask.sizes.items() <= ids.sizes.items():
        raise sc.DimensionError(
            f"Mask '{name}' has sizes {mask.sizes}, which are not part of the sizes "
            f"of the detector, {ids.sizes}."
        )
    # A mask file may use any order. DetectorMasks then has one layout for all of them.
    return mask.transpose([dim for dim in ids.dims if dim in mask.dims])
