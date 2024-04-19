from ess.spectroscopy.types import *


def sample_analyzer_vector(
        sample_position: SamplePosition,
        analyzer_position: AnalyzerPosition,
        analyzer_orientation: AnalyzerOrientation,
        detector_position: DetectorPosition
) -> SampleAnalyzerVector:
    """Determine the sample to analyzer-reflection-point vector per detector element

    :parameter sample_position: scipp.DType.vector3
        The (probably singular) sample position, typically (0, 0, 0)
    :parameter analyzer_position: scipp.DType.vector3
        The nominal center of the central analyzer blade *surface*
    :parameter analyzer_orientation: scipp.DType.rotate3
        The orienting quaternion of the analyzer, used to identify the crystal y-axis
    :parameter detector_position: scipp.DType.vector3
        The position of the detector element

    :Note:
    The shapes of the analyzer position and orientation should be self-consistent and will likely be 1:1.
    There is expected to be multiple detector element positions per analyzer which can be represented as
    an additional dimension compared to the analyzer shapes.
    """
    from scipp import concat, vector, dot, sqrt, DType
    from ess.spectroscopy.utils import norm

    # Scipp does not distinguish between coordinates and directions, so we need to do some extra legwork
    # to ensure we can apply the orientation transformation _and_ obtain a dimensionless direction vector
    y = vector([0, 1, 0], unit=analyzer_orientation.unit)

    yhat = (analyzer_orientation * vector([0, 1, 0], unit=analyzer_orientation.unit)
            - analyzer_orientation * vector([0, 0, 0], unit=analyzer_orientation.unit))
    yhat /= norm(yhat)

    sample_analyzer_center_vector = analyzer_position - sample_position

    sample_detector_vector = detector_position - sample_position
    sd_out_of_plane = dot(sample_detector_vector, yhat)
    # the sample-detector vector is the sum of sample-analyzer, analyzer-detector-center, the out-of-plane vector
    analyzer_detector_center_vector = sample_detector_vector - sample_analyzer_center_vector - sd_out_of_plane * yhat

    # TODO Consider requiring that dot(analyzer_position - sample_position, yhat) is zero?

    sample_analyzer_center_distance = norm(sample_analyzer_center_vector)

    analyzer_detector_center_distance = norm(analyzer_detector_center_vector)

    # similar-triangles give the out-of-plane analyzer reflection point distance
    sa_out_of_plane = sample_analyzer_center_distance / (sample_analyzer_center_distance + analyzer_detector_center_distance) * sd_out_of_plane

    return sample_analyzer_center_vector + sa_out_of_plane * yhat


def analyzer_detector_vector(
        sample_position: SamplePosition,
        sample_analyzer_vec: SampleAnalyzerVector,
        detector_position: DetectorPosition
) -> AnalyzerDetectorVector:
    return detector_position - (sample_position + sample_analyzer_vec)


def kf_hat(sample_analyzer_vec: SampleAnalyzerVector) -> SampleAnalyzerDirection:
    from ess.spectroscopy.utils import norm
    return sample_analyzer_vec / norm(sample_analyzer_vec)


def kf_wavenumber(
        sample_analyzer_vec: SampleAnalyzerVector,
        analyzer_detector_vec: AnalyzerDetectorVector,
        tau: ReciprocalLatticeSpacing | ReciprocalLatticeVectorAbsolute
) -> Wavenumber:
    from scipp import sqrt, DType
    from ess.spectroscopy.utils import norm
    if tau.dtype == DType.vector3:
        tau = norm(tau)

    # law of Cosines gives the scattering angle based on distances:
    l_sa = norm(sample_analyzer_vec)
    l_ad = norm(analyzer_detector_vec)
    l_diff = norm(sample_analyzer_vec - analyzer_detector_vec)
    # 2 theta is measured from the direction S-A, so the internal angle is (pi - 2 theta)
    # and the normal law of Cosines is modified accordingly to be -cos(2 theta) instead of cos(pi - 2 theta)
    cos2theta = (l_diff * l_diff - l_sa * l_sa - l_ad * l_ad) / (2 * l_sa * l_ad)

    # law of Cosines gives the Bragg reflected wavevector magnitude
    return tau / sqrt(2 - 2 * cos2theta)


def kf_vector(
        kf_direction: SampleAnalyzerDirection,
        kf_magnitude: Wavenumber
) -> Wavevector:
    return kf_direction * kf_magnitude


def secondary_flight_path_length(
        sample_analyzer_vec: SampleAnalyzerVector,
        analyzer_detector_vec: AnalyzerDetectorVector
) -> SampleDetectorPathLength:
    from ess.spectroscopy.utils import norm
    return norm(sample_analyzer_vec) + norm(analyzer_detector_vec)


def secondary_flight_time(
        secondary_flight_distance: SampleDetectorPathLength,
        kf_magnitude: Wavenumber
) -> SampleDetectorFlightTime:
    from scipp.constants import hbar, neutron_mass
    velocity = kf_magnitude * hbar / neutron_mass
    return secondary_flight_distance / velocity


def sample_time(
        detector_time: DetectorTime,
        secondary_time: SampleDetectorFlightTime
) -> SampleTime:
    return detector_time - secondary_time
