# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Pixel masks stored in files.

Reading is independent of runs and detector banks, so a workflow reads its mask files
once. Converting the file content to masks for a detector is a separate step.
"""

import re
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

_USER_DEFINED_MASK = 1 << 8
"""Bit of the NeXus ``pixel_mask`` field for a mask defined by the user."""

_REJECTING_BITS = 0xFFFF
"""Bits of the NeXus ``pixel_mask`` field that exclude a pixel from analysis.

Other bits are tags, which do not exclude a pixel on their own.
"""


def read_pixel_masks(filenames: PixelMaskFilenames) -> PixelMasks:
    """Read pixel masks from files.

    The following formats are supported, selected by file suffix and content:

    - ``.xml``: Masked detector IDs in the ISIS format,
      see :py:func:`read_xml_masked_detector_ids`.
    - ``.h5``, ``.hdf``, ``.hdf5``, ``.nxs``: Either of

      - A NeXus file with masks by detector bank, see :py:func:`write_pixel_masks`:

        .. code-block:: text

            /entry                         (NXentry)
                title                      name of the mask, optional
                instrument                 (NXinstrument)
                    loki_detector_0        (NXdetector)
                        pixel_mask         integer array

        Bank names are the names of the NeXus detector groups. A bank without
        ``pixel_mask`` has no mask from this file. Following NeXus, a pixel is masked
        if any bit in ``pixel_mask & 0xFFFF`` is set. The dimensions of ``pixel_mask``
        are given by HDF5 dimension labels. They are named after the logical dimensions
        of the bank, and may be a subset of them. Other entries of the file, such as a
        validity date, are ignored.
      - A file written by :py:func:`scipp.io.save_hdf5` with a variable of masked
        detector IDs.

    A mask has the name stored in its file. Otherwise it is named after its file,
    without the directory, since directories differ between hosts.

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
        name, content = _read_mask_file(filename)
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

    Masks given as detector IDs apply to any bank, since IDs identify a pixel. They
    have the dimensions of ``ids``. Masks given per bank are selected by ``bank``.

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
            masks[name] = sc.array(
                dims=ids.dims, values=np.isin(ids.values, content.values)
            )
            continue
        if bank is None:
            raise ValueError(f"Mask '{name}' is given per bank, but no bank is known.")
        if known_banks is not None and (unknown := set(content) - set(known_banks)):
            raise ValueError(f"Mask '{name}' has unknown banks: {sorted(unknown)}.")
        if bank in content:
            masks[name] = _validate_bank_mask(content[bank], ids, name)
    return masks


def write_pixel_masks(
    filename: str | Path,
    masks: Mapping[str, sc.Variable],
    name: str | None = None,
) -> None:
    """Write boolean masks by detector bank to a NeXus file.

    See :py:func:`read_pixel_masks` for the layout of the file.

    Parameters
    ----------
    filename:
        Path of the file to write.
    masks:
        Boolean mask of each detector bank. Dimensions are named after the logical
        dimensions of the bank, and may be a subset of them.
    name:
        Name of the mask. Without it, the mask is named after the file when it is read.
    """
    with snx.File(filename, 'w') as file:
        entry = file.create_class('entry', snx.NXentry)
        if name is not None:
            entry.create_field('title', name)
        instrument = entry.create_class('instrument', snx.NXinstrument)
        for bank, mask in masks.items():
            detector = instrument.create_class(bank, snx.NXdetector)
            bits = np.where(mask.values, _USER_DEFINED_MASK, 0)
            dataset = detector.create_field(
                'pixel_mask',
                sc.array(dims=mask.dims, values=bits, unit=None, dtype='int32'),
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


def _read_mask_file(filename: Path) -> tuple[str, sc.Variable | sc.DataGroup]:
    if filename.suffix == '.xml':
        return filename.name, read_xml_masked_detector_ids(filename)
    if filename.suffix in _HDF5_SUFFIXES:
        return _read_hdf5(filename)
    raise ValueError(f"Unsupported suffix of mask file '{filename}'.")


def _read_hdf5(filename: Path) -> tuple[str, sc.Variable | sc.DataGroup]:
    with snx.File(filename) as file:
        if 'entry' in file:
            return _read_nexus_masks(file['entry'], filename)
    ids = sc.io.load_hdf5(filename)
    if isinstance(ids, sc.Variable) and ids.dtype in ('int32', 'int64'):
        return filename.name, ids
    raise ValueError(
        f"Mask file '{filename}' holds neither a NeXus entry nor integer detector IDs."
    )


def _read_nexus_masks(entry: snx.Group, filename: Path) -> tuple[str, sc.DataGroup]:
    masks = {
        bank: _read_bank_mask(detector, filename)
        for bank, detector in entry['instrument'][snx.NXdetector].items()
        if 'pixel_mask' in detector
    }
    if not masks:
        raise ValueError(f"Mask file '{filename}' has no detector with a pixel_mask.")
    name = entry['title'][()] if 'title' in entry else filename.name
    return name, sc.DataGroup(masks)


def _read_bank_mask(detector: snx.Group, filename: Path) -> sc.Variable:
    # NeXus combines these into one mask. They are rejected instead of ignored, since
    # ignoring them would leave pixels unmasked.
    if numbered := [key for key in detector if re.fullmatch(r'pixel_mask_\d+', key)]:
        raise ValueError(
            f"Mask file '{filename}' has several masks for one bank: {numbered}."
        )
    field = detector['pixel_mask']
    # Loading the field as a variable would clip uint32 to int32 and change the bits.
    values = field.dataset[()]
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Mask file '{filename}' has a non-integer pixel_mask.")
    # int64 holds all bits to test for any input type, unlike e.g. uint8.
    masked = values.astype('int64') & _REJECTING_BITS != 0
    return sc.array(dims=field.dims, values=masked)


def _validate_bank_mask(mask: sc.Variable, ids: sc.Variable, name: str) -> sc.Variable:
    # Masks with a dimension that the data lacks would be ignored without an error.
    if not mask.sizes.items() <= ids.sizes.items():
        raise sc.DimensionError(
            f"Mask '{name}' has sizes {mask.sizes}, which are not part of the sizes "
            f"of the detector, {ids.sizes}."
        )
    return mask
