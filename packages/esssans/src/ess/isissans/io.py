# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
File loading functions for ISIS data, NOT using Mantid.
"""

from typing import NewType

import sciline
import scipp as sc

from ess.sans.types import (
    BackgroundRun,
    DirectBeam,
    DirectBeamFilename,
    Filename,
    MaskedDetectorIDs,
    PixelMaskFilename,
    RunType,
    SampleRun,
    TransmissionRun,
)

CalibrationFilename = NewType('CalibrationFilename', str)


def read_xml_detector_masking(filename: PixelMaskFilename) -> MaskedDetectorIDs:
    """Read a pixel mask from an XML file.

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


class LoadedFileContents(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Contents of a loaded file."""


def load_tutorial_run(filename: Filename[RunType]) -> LoadedFileContents[RunType]:
    return LoadedFileContents[RunType](sc.io.load_hdf5(filename))


def load_tutorial_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    return DirectBeam(sc.io.load_hdf5(filename))


def transmission_from_sample_run(
    data: LoadedFileContents[SampleRun],
) -> LoadedFileContents[TransmissionRun[SampleRun]]:
    """
    Use transmission from a sample run, instead of dedicated run.
    """
    return LoadedFileContents[TransmissionRun[SampleRun]](data)


def transmission_from_background_run(
    data: LoadedFileContents[BackgroundRun],
) -> LoadedFileContents[TransmissionRun[BackgroundRun]]:
    """
    Use transmission from a background run, instead of dedicated run.
    """
    return LoadedFileContents[TransmissionRun[BackgroundRun]](data)


providers = (read_xml_detector_masking,)
