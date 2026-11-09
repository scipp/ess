# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Pixel masks stored in files.

Reading is independent of runs and detector banks, so a workflow reads its mask files
once. Converting the file content to masks for a detector is a separate step.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import NewType

import numpy as np
import scipp as sc

PixelMaskFilenames = NewType('PixelMaskFilenames', Sequence[str | Path])
"""Files with pixel masks to apply. Use an empty sequence to apply none."""

LoadedPixelMasks = NewType('LoadedPixelMasks', dict[str, sc.Variable])
"""Masked detector IDs from each mask file, by mask name."""


def read_pixel_masks(filenames: PixelMaskFilenames) -> LoadedPixelMasks:
    """Read pixel masks from files.

    A mask is named after its file, without the directory, since directories differ
    between hosts. The following formats are supported, selected by file suffix:

    - ``.xml``: Masked detector IDs in the ISIS format,
      see :py:func:`read_xml_masked_detector_ids`.
    - ``.h5``, ``.hdf``, ``.hdf5``: A scipp variable of masked detector IDs, as written
      by :py:func:`scipp.io.save_hdf5`.

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
        if filename.name in masks:
            raise ValueError(f"Several mask files have the name '{filename.name}'.")
        masks[filename.name] = _read_mask_file(filename)
    return LoadedPixelMasks(masks)


def to_pixel_masks(
    loaded: LoadedPixelMasks, ids: sc.Variable
) -> dict[str, sc.Variable]:
    """Create a boolean mask per loaded mask, for the pixels of a detector.

    Parameters
    ----------
    loaded:
        Masks read from files.
    ids:
        Detector IDs of the pixels. The masks have the dimensions of this variable.
    """
    return {
        name: sc.array(dims=ids.dims, values=np.isin(ids.values, masked.values))
        for name, masked in loaded.items()
    }


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


def _read_mask_file(filename: Path) -> sc.Variable:
    if filename.suffix == '.xml':
        return read_xml_masked_detector_ids(filename)
    if filename.suffix in ('.h5', '.hdf', '.hdf5'):
        return _validate_ids(sc.io.load_hdf5(filename), filename)
    raise ValueError(f"Unsupported suffix of mask file '{filename}'.")


def _validate_ids(content: object, filename: Path) -> sc.Variable:
    if isinstance(content, sc.Variable) and content.dtype in ('int32', 'int64'):
        return content
    raise ValueError(f"Mask file '{filename}' does not hold integer detector IDs.")
