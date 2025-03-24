# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc
from scippneutron.conversion.beamline import (
    BeamAlignedUnitVectors,
)

from ess.spectroscopy.types import (
    Analyzer,
    DataAtSample,
    DataGroupedByRotation,
    PulsePeriod,
    RunType,
    SecondarySpecCoordTransformGraph,
)

from ..utils import in_same_unit


def sample_analyzer_vector(
    sample_position: sc.Variable,
    analyzer_position: sc.Variable,
    analyzer_transform: sc.Variable,
    detector_position: sc.Variable,
) -> sc.Variable:
    """Determine the sample to analyzer-reflection-point vector per detector element

    Note
    ----
    The shapes of the analyzer position and orientation should be self-consistent
    and will likely be 1:1. There is expected to be multiple detector element positions
    per analyzer which can be represented as an additional dimension compared to
    the analyzer shapes.

    Parameters
    ----------
    sample_position:
        The (probably scalar) sample position, typically vector (0, 0, 0).
    analyzer_position:
        The nominal center of the central analyzer blade *surface*.
    analyzer_transform:
        The transform that places and rotates the analyzer in lab space.
        Only the rotation is extracted to identify the crystal y-axis.
    detector_position:
        The position of the detector element.

    Returns
    -------
    :
        The vector from the sample position to the interaction point on the analyzer
        for each detector element.
    """
    # Scipp does not distinguish between coordinates and directions, so we need to do
    # some extra legwork to ensure we can apply the orientation transformation
    # _and_ obtain a dimensionless direction vector
    o = sc.vector([0, 0, 0], unit=analyzer_transform.unit)
    y = sc.vector(
        [0, 1, 0], unit=analyzer_transform.unit
    )  # and y perpendicular to the scattering plane
    yhat = analyzer_transform * y - analyzer_transform * o
    yhat /= sc.norm(yhat)

    sample_analyzer_center_vector = analyzer_position - sample_position

    sample_detector_vector = detector_position - sample_position
    sd_out_of_plane = sc.dot(sample_detector_vector, yhat)
    # the sample-detector vector is the sum of sample-analyzer,
    # analyzer-detector-center, the out-of-plane vector
    analyzer_detector_center_vector = (
        sample_detector_vector - sample_analyzer_center_vector - sd_out_of_plane * yhat
    )

    # TODO Consider requiring that dot(analyzer_position-sample_position, yhat) is zero?

    sample_analyzer_center_distance = sc.norm(sample_analyzer_center_vector)

    analyzer_detector_center_distance = sc.norm(analyzer_detector_center_vector)

    # similar-triangles give the out-of-plane analyzer reflection point distance
    sa_out_of_plane = (
        sample_analyzer_center_distance
        / (sample_analyzer_center_distance + analyzer_detector_center_distance)
        * sd_out_of_plane
    )

    return sample_analyzer_center_vector + sa_out_of_plane * yhat


def detector_geometric_a4(
    *,
    sample_analyzer_vector: sc.Variable,
    beam_aligned_unit_vectors: BeamAlignedUnitVectors,
) -> sc.Variable:
    """Calculate the scattering angle from the incident beam to each detector element

    Parameters
    ----------
    sample_analyzer_vector:
        The per detector element analyzer interaction position, as determined
        by :func:`sample_analyzer_vector`.
    beam_aligned_unit_vectors:
        Unit vectors defining the coordinate system aligned with the incident beam
        and gravity.
        See :func:`scippneutron.conversion.beamline.beam_aligned_unit_vectors`.

    Returns
    -------
    :
        The per detector element scattering angle, a4, in degrees.
    """
    lab_x = beam_aligned_unit_vectors['beam_aligned_unit_x']
    lab_z = beam_aligned_unit_vectors['beam_aligned_unit_z']
    x = sc.dot(lab_x, sample_analyzer_vector)
    z = sc.dot(lab_z, sample_analyzer_vector)
    return sc.atan2(y=x, x=z).to(unit='deg')


def analyzer_detector_vector(
    sample_position: sc.Variable,
    sample_analyzer_vector: sc.Variable,
    detector_position: sc.Variable,
) -> sc.Variable:
    """Calculate the analyzer-detector vector"""
    return detector_position - (sample_position + sample_analyzer_vector)


def final_wavenumber(
    sample_analyzer_vector: sc.Variable,
    analyzer_detector_vector: sc.Variable,
    analyzer_dspacing: sc.Variable,
) -> sc.Variable:
    r"""Find the wave number of the neutrons reflected to each detector-element.

    The wave number and wavelength are inversely proportional with

    .. math::

        \text{wavelength} = 2 * \pi / \text{wavenumber}

    Parameters
    ----------
    sample_analyzer_vector:
        The vector from the sample to the analyzer interaction point for each
        detector element.
    analyzer_detector_vector:
        The vector from the analyzer interaction point to its detector element,
        for each detector element.
    analyzer_dspacing:
        The lattice plane spacing of the analyzer crystal.

    Returns
    -------
    :
        The magnitude of the reflected neutron wave vector for each detector element.
    """
    # law of Cosines gives the scattering angle based on distances:
    l_sa = sc.norm(sample_analyzer_vector)
    l_ad = sc.norm(analyzer_detector_vector)
    l_diff = sc.norm(sample_analyzer_vector + analyzer_detector_vector)
    # 2 theta is measured from the direction S-A, so the internal angle is
    # (pi - 2 theta) and the normal law of Cosines is modified accordingly to be
    # -cos(2 theta) instead of cos(pi - 2 theta)
    cos2theta = (l_diff * l_diff - l_sa * l_sa - l_ad * l_ad) / (2 * l_sa * l_ad)

    # law of Cosines gives the Bragg reflected wavevector magnitude
    return 2 * np.pi / analyzer_dspacing / sc.sqrt(2 - 2 * cos2theta)


def final_energy(final_wavenumber: sc.Variable) -> sc.Variable:
    """Converts (final) wave number to (final) energy"""
    from scipp.constants import hbar, neutron_mass

    return ((hbar * hbar / 2 / neutron_mass) * final_wavenumber * final_wavenumber).to(
        unit='meV'
    )


def final_wavevector(
    sample_analyzer_vector: sc.Variable, final_wavenumber: sc.Variable
) -> sc.Variable:
    """Constructs the final wave vector form its direction and magnitude"""
    return sample_analyzer_vector / sc.norm(sample_analyzer_vector) * final_wavenumber


def secondary_flight_path_length(
    sample_analyzer_vector: sc.Variable,
    analyzer_detector_vector: sc.Variable,
) -> sc.Variable:
    """Returns the path-length-distance between the sample and each detector element"""
    return sc.norm(sample_analyzer_vector) + sc.norm(analyzer_detector_vector)


def secondary_flight_time(
    L2: sc.Variable, final_wavenumber: sc.Variable
) -> sc.Variable:
    """Calculates the most-likely time-of-flight between the sample and each pixel"""
    from scipp.constants import hbar, neutron_mass

    velocity = final_wavenumber * (hbar / neutron_mass)
    return sc.to_unit(L2 / velocity, 'ms', copy=False)


def secondary_spectrometer_coordinate_transformation_graph(
    analyzer: Analyzer[RunType],
) -> SecondarySpecCoordTransformGraph[RunType]:
    """Return a coordinate transformation graph for the secondary spectrometer.

    Parameters
    ----------
    analyzer:
        Data group with analyzer parameters.

    Returns
    -------
    :
        Coordinate transformation graph for the secondary spectrometer.
        The graph captures the relevant parameters of ``analyzer``.
    """
    from scippneutron.conversion.beamline import beam_aligned_unit_vectors

    return SecondarySpecCoordTransformGraph[RunType](
        {
            "a4": detector_geometric_a4,
            "analyzer_dspacing": lambda: analyzer["dspacing"],
            "analyzer_position": lambda: analyzer["position"],
            "analyzer_transform": lambda: analyzer["transform"],
            "beam_aligned_unit_vectors": beam_aligned_unit_vectors,
            "detector_position": "position",
            "sample_analyzer_vector": sample_analyzer_vector,
            "analyzer_detector_vector": analyzer_detector_vector,
            "final_energy": final_energy,
            "final_wavenumber": final_wavenumber,
            "final_wavevector": final_wavevector,
            "L2": secondary_flight_path_length,
            "secondary_flight_time": secondary_flight_time,
        }
    )


def move_time_to_sample(
    data: DataGroupedByRotation[RunType], pulse_period: PulsePeriod
) -> DataAtSample[RunType]:
    """Return the events with the event_time_offset coordinate offset to time at sample.

    Parameters
    ----------
    data:
        Data array with "event_time_offset" and "secondary_flight_time" coordinates.
    pulse_period:
        Duration of a neutron pulse.

    Returns
    -------
    :
        A shallow copy of ``data`` where the "event_time_offset" coordinate has been
        shifted to the time at the sample.
    """
    offset = in_same_unit(
        data.coords['secondary_flight_time'], data.bins.coords['event_time_offset']
    )
    time = data.bins.coords['event_time_offset'] - offset
    time %= in_same_unit(pulse_period, time)
    return DataAtSample[RunType](
        data
        # These are the detector positions and they no longer match the time:
        .drop_coords(('position', 'x_pixel_offset', 'y_pixel_offset'))
        # Use bins.assign_coords to avoid modifying the original data:
        .bins.assign_coords(event_time_offset=time)
    )


providers = (
    secondary_spectrometer_coordinate_transformation_graph,
    move_time_to_sample,
)
