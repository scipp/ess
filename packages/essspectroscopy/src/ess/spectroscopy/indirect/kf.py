from ess.spectroscopy.types import (
    AnalyzerDetectorVector,
    AnalyzerOrientation,
    AnalyzerPosition,
    DetectorFrameTime,
    DetectorGeometricA4,
    DetectorPosition,
    FinalEnergy,
    FinalWavenumber,
    FinalWavevector,
    ReciprocalLatticeSpacing,
    ReciprocalLatticeVectorAbsolute,
    SampleAnalyzerDirection,
    SampleAnalyzerVector,
    SampleDetectorFlightTime,
    SampleDetectorPathLength,
    SampleFrameTime,
    SamplePosition,
)


def sample_analyzer_vector(
    sample_position: SamplePosition,
    analyzer_position: AnalyzerPosition,
    analyzer_orientation: AnalyzerOrientation,
    detector_position: DetectorPosition,
) -> SampleAnalyzerVector:
    """Determine the sample to analyzer-reflection-point vector per detector element

    Note
    ----
    The shapes of the analyzer position and orientation should be self-consistent
    and will likely be 1:1. There is expected to be multiple detector element positions
    per analyzer which can be represented as an additional dimension compared to
    the analyzer shapes.

    Parameters
    ----------
    sample_position: scipp.DType.vector3
        The (probably singular) sample position, typically (0, 0, 0)
    analyzer_position: scipp.DType.vector3
        The nominal center of the central analyzer blade *surface*
    analyzer_orientation: scipp.DType.rotate3
        The orienting quaternion of the analyzer, used to identify the crystal y-axis
    detector_position: scipp.DType.vector3
        The position of the detector element

    Returns
    -------
    :
        The vector from the sample position to the interaction point on the analyzer
        for each detector element
    """
    from scipp import dot, vector

    from ess.spectroscopy.utils import norm

    # Scipp does not distinguish between coordinates and directions, so we need to do
    # some extra legwork to ensure we can apply the orientation transformation
    # _and_ obtain a dimensionless direction vector
    o = vector([0, 0, 0], unit=analyzer_orientation.unit)
    y = vector(
        [0, 1, 0], unit=analyzer_orientation.unit
    )  # and y perpendicular to the scattering plane
    yhat = analyzer_orientation * y - analyzer_orientation * o
    yhat /= norm(yhat)

    sample_analyzer_center_vector = analyzer_position - sample_position

    sample_detector_vector = detector_position - sample_position
    sd_out_of_plane = dot(sample_detector_vector, yhat)
    # the sample-detector vector is the sum of sample-analyzer,
    # analyzer-detector-center, the out-of-plane vector
    analyzer_detector_center_vector = (
        sample_detector_vector - sample_analyzer_center_vector - sd_out_of_plane * yhat
    )

    # TODO Consider requiring that dot(analyzer_position-sample_position, yhat) is zero?

    sample_analyzer_center_distance = norm(sample_analyzer_center_vector)

    analyzer_detector_center_distance = norm(analyzer_detector_center_vector)

    # similar-triangles give the out-of-plane analyzer reflection point distance
    sa_out_of_plane = (
        sample_analyzer_center_distance
        / (sample_analyzer_center_distance + analyzer_detector_center_distance)
        * sd_out_of_plane
    )

    return sample_analyzer_center_vector + sa_out_of_plane * yhat


def detector_geometric_a4(vec: SampleAnalyzerVector) -> DetectorGeometricA4:
    """Calculate the scattering angle from the incident beam to each detector element

    Parameters
    ----------
    vec : scipp.DType.vector3
        The per detector element analyzer interaction position, as determined
        by `sample_analyzer_vector`

    Returns
    -------
    :
        The per detector element scattering angle, a4, in degrees.
    """
    from scipp import atan2, dot, vector

    lab_x = vector(
        [1, 0, 0]
    )  # perpendicular to the incident beam, in the horizontal plane
    lab_z = vector([0, 0, 1])  # along the incident beam direction
    return atan2(y=dot(lab_x, vec), x=dot(lab_z, vec)).to(unit='deg')


def analyzer_detector_vector(
    sample_position: SamplePosition,
    sample_analyzer_vec: SampleAnalyzerVector,
    detector_position: DetectorPosition,
) -> AnalyzerDetectorVector:
    """Calculate the analyzer-detector vector"""
    return detector_position - (sample_position + sample_analyzer_vec)


def kf_hat(sample_analyzer_vec: SampleAnalyzerVector) -> SampleAnalyzerDirection:
    """Calculate the direction of the neutrons for each detector-element"""
    from ess.spectroscopy.utils import norm

    return sample_analyzer_vec / norm(sample_analyzer_vec)


def reciprocal_lattice_spacing(tau_vector: ReciprocalLatticeVectorAbsolute):
    """Calculate the distance between lattice planes in, e.g., the analyzer"""
    from ess.spectroscopy.utils import norm

    return norm(tau_vector)


def final_wavenumber(
    sample_analyzer_vec: SampleAnalyzerVector,
    analyzer_detector_vec: AnalyzerDetectorVector,
    tau: ReciprocalLatticeSpacing,
) -> FinalWavenumber:
    """Find the wave number of the neutrons reflected to each detector-element

    The wave number and wavelength are inversely proportional with
        wavelength = 2 * pi / wavenumber

    Parameters
    ----------
    sample_analyzer_vec : scipp.DType.vector3
        The vector from the sample to the analyzer interaction point for each
         detector element
    analyzer_detector_vec: scipp.DType.vector3
        The vector from the analyzer interaction point to its detector element,
        for each detector element
    tau: float-like
        The reciprocal lattice plane spacing of the analyzer crystal,
        likely in inverse angstrom

    Returns
    -------
    :
        The magnitude of the reflected neutron wave vector for each detector element
    """
    from scipp import sqrt

    from ess.spectroscopy.utils import norm

    # law of Cosines gives the scattering angle based on distances:
    l_sa = norm(sample_analyzer_vec)
    l_ad = norm(analyzer_detector_vec)
    l_diff = norm(sample_analyzer_vec + analyzer_detector_vec)
    # 2 theta is measured from the direction S-A, so the internal angle is
    # (pi - 2 theta) and the normal law of Cosines is modified accordingly to be
    # -cos(2 theta) instead of cos(pi - 2 theta)
    cos2theta = (l_diff * l_diff - l_sa * l_sa - l_ad * l_ad) / (2 * l_sa * l_ad)

    # law of Cosines gives the Bragg reflected wavevector magnitude
    return tau / sqrt(2 - 2 * cos2theta)


def final_energy(kf: FinalWavenumber) -> FinalEnergy:
    """Converts (final) wave number to (final) energy"""
    from scipp.constants import hbar, neutron_mass

    return ((hbar * hbar / 2 / neutron_mass) * kf * kf).to(unit='meV')


def final_wavevector(
    kf_direction: SampleAnalyzerDirection, kf_magnitude: FinalWavenumber
) -> FinalWavevector:
    """Constructs the final wave vector form its direction and magnitude"""
    return kf_direction * kf_magnitude


def secondary_flight_path_length(
    sample_analyzer_vec: SampleAnalyzerVector,
    analyzer_detector_vec: AnalyzerDetectorVector,
) -> SampleDetectorPathLength:
    """Returns the path-length-distance between the sample and each detector element"""
    from ess.spectroscopy.utils import norm

    return norm(sample_analyzer_vec) + norm(analyzer_detector_vec)


def secondary_flight_time(
    secondary_flight_distance: SampleDetectorPathLength, kf_magnitude: FinalWavenumber
) -> SampleDetectorFlightTime:
    """Calculates the most-likely time-of-flight between the sample and each pixel"""
    from scipp.constants import hbar, neutron_mass

    velocity = kf_magnitude * hbar / neutron_mass
    return secondary_flight_distance / velocity


def sample_frame_time(
    detector_time: DetectorFrameTime, secondary_time: SampleDetectorFlightTime
) -> SampleFrameTime:
    """Return the time each neutron likely interacted with the sample"""
    return detector_time - secondary_time


providers = (
    sample_analyzer_vector,
    analyzer_detector_vector,
    kf_hat,
    final_wavenumber,
    final_wavevector,
    secondary_flight_path_length,
    secondary_flight_time,
    sample_frame_time,
    final_energy,
    detector_geometric_a4,
)
