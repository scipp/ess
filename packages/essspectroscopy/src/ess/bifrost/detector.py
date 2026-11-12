# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Detector handling for BIFROST."""

from collections.abc import Callable

import numpy as np
import scipp as sc
import scippnexus as snx
from ess.spectroscopy.indirect.conversion import add_spectrometer_coords
from ess.spectroscopy.types import (
    Analyzer,
    Analyzers,
    DetectorPositionOffset,
    EmptyDetector,
    NeXusComponent,
    NeXusMultiComponent,
    NeXusMultiTransformation,
    NeXusTransformation,
    PrimarySpecCoordTransformGraph,
    RunType,
    SecondarySpecCoordTransformGraph,
)

from ess.reduce.nexus.types import Position

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


def single_arc_and_channel_from_detector_number(
    detector_number: sc.Variable,
) -> tuple[sc.Variable, sc.Variable]:
    """Calculate arc number and channel from the detector number of a single triplet.

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
    detector_components: NeXusMultiComponent[snx.NXdetector, RunType],
    analyzers: Analyzers[RunType],
    *,
    primary_graph: PrimarySpecCoordTransformGraph[RunType],
    secondary_graph: SecondarySpecCoordTransformGraph[RunType],
    transforms: NeXusMultiTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> EmptyDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    This includes:

    - Reshaping the data array to the logical detector shape.
    - Assigning geometry coordinate "position".
    - Assigning spectrometer coordinates such as "final_energy",
      "secondary_flight_time", and "L1".

    Parameters
    ----------
    detector_components:
        Loaded NeXus detector.
    analyzers:
        Loaded analyzer parameters.
    primary_graph:
        Coordinate transformation graph for the primary spectrometer.
    secondary_graph:
        Coordinate transformation graph for the secondary spectrometer.
        Must be a closure over analyzer parameters.
        And those parameters must have a compatible shape with ``data``.
    transforms:
        NeXus transformations for the detectors.
    offset:
        Offset to add to the detector position.

    Returns
    -------
    :
        Detector geometry and spectrometer coordinates.
    """
    entries = []
    for analyzer in analyzers.values():
        detector_name = analyzer['detector_name']
        da = get_single_calibrated_detector_bifrost(
            detector_components[detector_name],
            transform=transforms.value[detector_name],
            offset=offset,
        )
        da = da.rename(dim_0='tube', dim_1='length')

        arc, channel = single_arc_and_channel_from_detector_number(
            da.coords['detector_number']
        )
        entries.append(
            (arc.value, channel.value, da, _extract_analyzer_parameters(da, analyzer))
        )

    entries.sort(key=lambda t: t[:2])  # sort by (arc, channel)
    arcs, channels, dets, analyzer_params = zip(*entries, strict=True)
    detector = sc.concat(dets, dim='triplet')
    if len(dets) == 1:
        # concat does not introduce a 'triplet' dim when there is only 1 detector.
        # But downstream code requires these coords to have that dim.
        for key in ('detector_number', 'x_pixel_offset', 'y_pixel_offset', 'position'):
            coord = detector.coords[key]
            detector.coords[key] = coord.broadcast(
                dims=['triplet', *coord.dims], shape=[1, *coord.shape]
            )

    detector = add_spectrometer_coords(
        detector,
        primary_graph,
        SecondarySpecCoordTransformGraph[RunType](
            {
                **secondary_graph,
                **_build_analyzer_coord_transform_graph(analyzer_params),
            }
        ),
    )
    return _fold_detector_to_arc_and_channel(detector, arcs, channels)


def _fold_detector_to_arc_and_channel(
    detector: sc.DataArray, arcs: list[int], channels: list[int]
) -> sc.DataArray:
    """Fold an empty detector's triplet dim to (arc, channel) if possible.

    The input arcs and channels must be in the same order as the detector's triplets.

    Modifies the input detector!
    """
    unique_arcs = np.unique(arcs)
    unique_channels = np.unique(channels)
    expected_pairs = [
        (arc, channel) for arc in unique_arcs for channel in unique_channels
    ]
    actual_pairs = list(zip(arcs, channels, strict=True))
    if actual_pairs == expected_pairs:
        # We have a rectangular subset of triplets -> fold them.
        n_arcs = len(unique_arcs)
        n_channels = len(unique_channels)
        folded = detector.fold('triplet', sizes={'arc': n_arcs, 'channel': n_channels})
        folded.coords['arc'] = sc.array(
            dims=['arc'], values=arcs[::n_channels], unit=None
        )
        folded.coords['channel'] = sc.array(
            dims=['channel'], values=channels[:n_channels], unit=None
        )

        # Transpose to make `detector_number` contiguous:
        detector_dims = ['arc', 'tube', 'channel', 'length']
        target_dims = [
            *(d for d in folded.dims if d not in detector_dims),
            *detector_dims,
        ]
        return folded.transpose(target_dims).copy()
    else:
        # Not a rectangular subset, cannot fold.
        detector.coords['arc'] = sc.array(dims=['triplet'], values=arcs, unit=None)
        detector.coords['channel'] = sc.array(
            dims=['triplet'], values=channels, unit=None
        )
        return detector


def get_single_calibrated_detector_bifrost(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> sc.DataArray:
    """Extract the data array corresponding to a detector's signal field.

    This function is specific to BIFROST and differs from the generic
    :func:`ess.reduce.nexus.workflow.get_calibrated_detector` in that it
    assigns time-dependent positions by broadcasting the data into the 'time' dimension.

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
        Detector with geometry coordinates.
    """

    from ess.reduce.nexus import compute_detector_position, extract_signal_data_array

    da = extract_signal_data_array(detector)
    # BIFROST's detectors ride the rotating tank, so a lab-frame offset added after
    # the transform cannot express any correction one would actually want; the
    # transformation chain is the handle. compute_detector_position requires the
    # argument, so pin it to zero here.
    position = compute_detector_position(da, transform=transform, offset=offset)
    return _assign_detector_position(da, position)


def _assign_detector_position(
    da: sc.DataArray, position: sc.Variable | sc.DataArray
) -> sc.DataArray:
    if isinstance(position, sc.DataArray):  # time-dependent transform
        # Store position and time as separate coords because we can't store data arrays.
        return da.broadcast(
            dims=['time', *da.dims], shape=[position.sizes['time'], *da.shape]
        ).assign_coords(position=position.data, time=position.coords['time'])
    return da.assign_coords(position=position)


# We insert the analyzer coords into the graph so that they don't end up as coords
# in the output. This could be done in the provider of SecondarySpecCoordTransformGraph
# but that provider would then have to request the detector component to check
# the time coordinates.
def _build_analyzer_coord_transform_graph(
    analyzer_parameters: list[dict[str, sc.Variable]],
) -> dict[str, Callable[..., sc.Variable]]:
    """Build a coord transformation graph that binds the analyzer parameters.

    The input should contain parameters for all relevant analyzers.
    *In the same order as detector banks.*
    """

    # can't use lambda in dict comprehension because it would bind to dynamic `key`
    def _just(value):
        def impl():
            return value

        return impl

    return {
        key: _just(sc.concat([ap[key] for ap in analyzer_parameters], dim='triplet'))
        for key in analyzer_parameters[0]
    }


def _extract_analyzer_parameters(
    detector: sc.DataArray,
    analyzer: Analyzer[RunType],
) -> dict[str, sc.Variable]:
    """Get the analyzer parameters needed for coordinate transforms of EmptyDetector."""
    ana_pos: Position[snx.NXcrystal, RunType] = analyzer['position']
    if ana_pos.is_dynamic:
        if 'time' not in detector.coords:
            raise sc.CoordError(
                "The analyzer position is time-dependent but the detector is not"
            )
        analyzer_positions = ana_pos.positions
        if not sc.identical(analyzer_positions.coords['time'], detector.coords['time']):
            raise sc.CoordError(
                f"The analyzer and detector positions are not at the same times.\n"
                f"Analyzer: {analyzer_positions.coords['time']}\n"
                f"Detector: {detector.coords['time']}\n"
                "This is likely due to a change in the NeXus structure. It used to "
                "guarantee that the times are identical."
            )
        analyzer_position = analyzer_positions.data
        analyzer_transform = analyzer['transform'].value.data
    else:
        analyzer_position = ana_pos.position
        analyzer_transform = analyzer['transform'].value

    return {
        'analyzer_dspacing': analyzer['dspacing'],
        'analyzer_position': analyzer_position,
        'analyzer_transform': analyzer_transform,
    }


providers = (
    arc_number,
    get_calibrated_detector_bifrost,
)
