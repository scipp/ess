# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx
from scippnexus.application_definitions import nxcansas

from .types import (
    BackgroundSubtractedIofQ,
    MaskedDetectorIDs,
    Measurement,
    OutFilename,
    PixelMaskFilename,
)


def save_background_subtracted_iofq(
    *,
    iofq: BackgroundSubtractedIofQ,
    out_filename: OutFilename,
    measurement: Measurement,
) -> None:
    """Save background-subtracted I(Q) histogram as an NXcanSAS file."""
    if iofq.bins is None:
        da = iofq.copy(deep=False)
    else:
        da = iofq.hist()
    if da.coords.is_edges('Q'):
        da.coords['Q'] = sc.midpoints(da.coords['Q'])
    with snx.File(out_filename, 'w') as f:
        f['sasentry'] = nxcansas.SASentry(
            title=measurement.title, run=measurement.run_number_maybe_int
        )
        f['sasentry']['sasdata'] = nxcansas.SASdata(da, Q_variances='resolutions')


def read_xml_detector_masking(filename: PixelMaskFilename) -> MaskedDetectorIDs:
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

    return MaskedDetectorIDs(
        sc.array(dims=['detector_id'], values=masked_detids, unit=None, dtype='int32')
    )
