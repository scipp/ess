# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Detector handling for BIFROST."""

import scipp as sc
import scippnexus as snx

from ess.spectroscopy.indirect.conversion import add_spectrometer_coords
from ess.spectroscopy.types import (
    DetectorPositionOffset,
    EmptyDetector,
    NeXusComponent,
    NeXusTransformation,
    PrimarySpecCoordTransformGraph,
    RunType,
    SecondarySpecCoordTransformGraph,
)

from .types import ArcNumber


def arc_number(
    beamline: EmptyDetector[RunType],
) -> ArcNumber[RunType]:
    """Calculate BIFROST arc index number from pixel final energy

    The BIFROST analyzers are each set to diffract an
    energy in the set (2.7, 3.2, 3.8, 4.4. 5.0) meV.
    This energy is only valid for the central point of the center
    tube of the associated detector triplet. All other pixels
    will have a final energy slightly higher or lower.

    This function assigns the closest arc number indexing the
    ordered set above.

    Parameters
    ----------
    beamline:
        A data array with a 'final_energy' coordinate which is the
        per-pixel (or event) final neutron energy.

    Returns
    -------
    :
        The arc index of the analyzer from which the neutron scattered
    """
    minimum = sc.scalar(2.7, unit='meV')
    step = sc.scalar(0.575, unit='meV')
    final_energy = beamline.coords['final_energy']
    return ArcNumber[RunType](sc.round((final_energy - minimum) / step).to(dtype='int'))


def arc_and_channel_from_detector_number(
    detector_number: sc.Variable,
) -> tuple[sc.Variable, sc.Variable]:
    """Calculate arc number and channel from detector number.

    Calculate arc and channel for this triplet based on detector_number layout.
    BIFROST detector_number ordering is (arc, tube, channel, pixel).
    Each triplet contains 3 tubes of 100 pixels for a single arc-channel pair.
    """

    det_num = detector_number['tube', 0]['length', 0].value
    pixels_per_tube = 100
    tubes_per_channel = 3
    channels_per_arc = 9
    pixels_per_arc = pixels_per_tube * tubes_per_channel * channels_per_arc  # 2700

    # detector_number is 1-indexed
    idx = det_num - 1
    arc = idx // pixels_per_arc
    remainder = idx % pixels_per_arc
    channel = (remainder % (pixels_per_tube * channels_per_arc)) // pixels_per_tube

    return sc.index(arc), sc.index(channel)


def get_calibrated_detector_bifrost(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
    primary_graph: PrimarySpecCoordTransformGraph[RunType],
    secondary_graph: SecondarySpecCoordTransformGraph[RunType],
) -> EmptyDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    The data array is reshaped to the logical detector shape.

    This function is specific to BIFROST and differs from the generic
    :func:`ess.reduce.nexus.workflow.get_calibrated_detector` in that it does not
    fold the detectors into logical dimensions because the files already contain
    the detectors in the correct shape.

    Parameters
    ----------
    detector:
        Loaded NeXus detector.
    transform:
        Transformation that determines the detector position.
    offset:
        Offset to add to the detector position.
    primary_graph:
        Coordinate transformation graph for the primary spectrometer.
    secondary_graph:
        Coordinate transformation graph for the secondary spectrometer.
        Must be a closure over analyzer parameters.
        And those parameters must have a compatible shape with ``data``.

    Returns
    -------
    :
        Detector geometry and spectrometer coordinates.
        This includes "final_energy", "secondary_flight_time", and "L1".
    """

    from ess.reduce.nexus.types import DetectorBankSizes
    from ess.reduce.nexus.workflow import get_calibrated_detector

    da = get_calibrated_detector(
        detector=detector,
        transform=transform,
        offset=offset,
        # The detectors are folded in the file, no need to do that here.
        bank_sizes=DetectorBankSizes({}),
    )
    da = da.rename(dim_0='tube', dim_1='length')

    arc, channel = arc_and_channel_from_detector_number(da.coords['detector_number'])
    da.coords['arc'] = arc
    da.coords['channel'] = channel

    da = add_spectrometer_coords(da, primary_graph, secondary_graph)

    return EmptyDetector[RunType](da)


def merge_triplets(
    *triplets: sc.DataArray,
) -> sc.DataArray:
    """Merge BIFROST detector triplets into a single data array.

    This function folds the triplets into (arc, channel) dimensions based on
    the scalar 'arc' and 'channel' coordinates assigned to each triplet. If the
    triplets form a regular rectangular subset of the full 5x9 detector array,
    they will be folded into those dimensions. Otherwise, they are concatenated
    along a 'triplet' dimension.

    Parameters
    ----------
    triplets:
        Data arrays to merge. Each must have scalar 'arc' and 'channel' coordinates.

    Returns
    -------
    :
        Input data arrays either folded into (arc, channel) dimensions or
        stacked along the "triplet" dimension.
    """
    if len(triplets) == 0:
        raise ValueError("At least one triplet is required")

    # Extract arc and channel from scalar coordinates
    arc_channel_pairs = [
        (triplet.coords['arc'].value, triplet.coords['channel'].value)
        for triplet in triplets
    ]

    # Sort triplets by (arc, channel)
    sorted_indices = sorted(range(len(triplets)), key=lambda i: arc_channel_pairs[i])
    sorted_triplets = [triplets[i] for i in sorted_indices]
    sorted_pairs = [arc_channel_pairs[i] for i in sorted_indices]

    # Check if the pairs form a regular rectangular grid
    unique_arcs = sorted({pair[0] for pair in sorted_pairs})
    unique_channels = sorted({pair[1] for pair in sorted_pairs})

    # Check if we have a complete rectangular subset
    expected_pairs = [
        (arc, channel) for arc in unique_arcs for channel in unique_channels
    ]
    concatenated = sc.concat(sorted_triplets, dim='triplet')
    if sorted_pairs == expected_pairs:
        # We have a regular grid, fold it
        return concatenated.fold(
            dim='triplet',
            sizes={'arc': len(unique_arcs), 'channel': len(unique_channels)},
        )

    # Fall back to simple concatenation if not a regular grid
    return concatenated


providers = (arc_number, get_calibrated_detector_bifrost)
