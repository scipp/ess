# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Detector handling for BIFROST."""

import scipp as sc
import scippnexus as snx

from ess.spectroscopy.types import (
    BeamlineWithSpectrometerCoords,
    CalibratedDetector,
    DetectorPositionOffset,
    NeXusComponent,
    NeXusTransformation,
    RunType,
)

from .types import ArcNumber


def arc_number(
    beamline: BeamlineWithSpectrometerCoords[RunType],
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


def get_calibrated_detector_bifrost(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> CalibratedDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector.
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

    Returns
    -------
    :
        Detector data.
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
    return CalibratedDetector[RunType](da)


def merge_triplets(
    *triplets: sc.DataArray,
) -> sc.DataArray:
    """Merge BIFROST detector triplets into a single data array.

    This function attempts to fold the triplets into (arc, channel) dimensions
    based on the detector_number coordinate. If the triplets form a regular
    rectangular subset of the full 5x9 detector array, they will be folded
    into those dimensions. Otherwise, they are concatenated along a 'triplet'
    dimension.

    Parameters
    ----------
    triplets:
        Data arrays to merge.

    Returns
    -------
    :
        Input data arrays either folded into (arc, channel) dimensions or
        stacked along the "triplet" dimension.
    """
    if len(triplets) == 0:
        raise ValueError("At least one triplet is required")

    # Extract detector_number from first pixel of first tube in each triplet
    # to determine arc and channel indices
    detector_numbers = []
    for triplet in triplets:
        if 'detector_number' not in triplet.coords:
            # If no detector_number, fall back to simple concatenation
            return sc.concat(triplets, dim="triplet")
        # Get the first detector number (position [0, 0] in tube, length dims)
        det_num = triplet.coords['detector_number']['tube', 0]['length', 0]
        detector_numbers.append(det_num.value)

    # Calculate arc and channel indices based on detector_number layout
    # BIFROST has 5 arcs, each with 9 channels, each with 3 tubes of 100 pixels
    # detector_number runs 1 to 13500, folded as (arc=5, tube=3, channel=9, pixel=100)
    # Layout within an arc:
    # - pixels 0-99: channel 0, tube 0
    # - pixels 100-199: channel 1, tube 0
    # - ...
    # - pixels 800-899: channel 8, tube 0
    # - pixels 900-999: channel 0, tube 1
    # - etc.
    pixels_per_tube = 100
    tubes_per_channel = 3
    channels_per_arc = 9
    pixels_per_arc = pixels_per_tube * tubes_per_channel * channels_per_arc  # 2700

    arc_channel_pairs = []
    for det_num in detector_numbers:
        # detector_number is 1-indexed
        idx = det_num - 1
        arc = idx // pixels_per_arc
        remainder = idx % pixels_per_arc
        # Within an arc, pattern repeats every (pixels_per_tube * channels_per_arc)
        # for each of the 3 tubes. Channel is determined by position within that.
        channel = (remainder % (pixels_per_tube * channels_per_arc)) // pixels_per_tube
        arc_channel_pairs.append((arc, channel))

    # Sort triplets by (arc, channel)
    sorted_indices = sorted(range(len(triplets)), key=lambda i: arc_channel_pairs[i])
    sorted_triplets = [triplets[i] for i in sorted_indices]
    sorted_pairs = [arc_channel_pairs[i] for i in sorted_indices]

    # Check if the pairs form a regular rectangular grid
    arcs = [pair[0] for pair in sorted_pairs]
    channels = [pair[1] for pair in sorted_pairs]

    unique_arcs = sorted(set(arcs))
    unique_channels = sorted(set(channels))

    # Check if we have a complete rectangular subset
    expected_count = len(unique_arcs) * len(unique_channels)
    if len(sorted_triplets) == expected_count:
        # Verify that all (arc, channel) combinations are present
        expected_pairs = [
            (arc, channel) for arc in unique_arcs for channel in unique_channels
        ]
        if sorted_pairs == expected_pairs:
            # We have a regular grid, fold it
            concatenated = sc.concat(sorted_triplets, dim='triplet')
            try:
                folded = concatenated.fold(
                    dim='triplet',
                    sizes={'arc': len(unique_arcs), 'channel': len(unique_channels)},
                )
                # Remove size-1 dimensions (e.g., if only one channel is present)
                folded = folded.squeeze()
                return folded
            except Exception as e:
                # If folding fails for any reason, fall back to triplet dim
                import warnings

                warnings.warn(
                    f"Failed to fold triplets into (arc, channel): {e}",
                    stacklevel=2,
                )
                pass

    # Fall back to simple concatenation if not a regular grid
    return sc.concat(triplets, dim="triplet")


providers = (arc_number, get_calibrated_detector_bifrost)
