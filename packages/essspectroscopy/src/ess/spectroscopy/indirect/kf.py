from scipp import Variable


def _norm(vector: Variable) -> Variable:
    from scipp import sqrt, dot, DType
    assert vector.dtype == DType.vector3  # "Vector operations require scipp.DType.vector3 elements!"
    return sqrt(dot(vector, vector))


def sample_analyzer_vector(
        sample_position: Variable,
        analyzer_position: Variable,
        analyzer_orientation: Variable,
        detector_position: Variable
) -> Variable:
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

    yhat = analyzer_orientation * vector([0, 1, 0])
    if analyzer_orientation.dtype != DType.rotation3:
        # If the orientation provided is a Affine transformation we need to subtract the coordinate translation
        yhat -= analyzer_orientation * vector([0, 0, 0])

    sample_analyzer_center_vector = analyzer_position - sample_position

    sample_detector_vector = detector_position - sample_position
    sd_out_of_plane = dot(sample_detector_vector, yhat)
    # the sample-detector vector is the sum of sample-analyzer, analyzer-detector-center, the out-of-plane vector
    analyzer_detector_center_vector = sample_detector_vector - sample_analyzer_center_vector - sd_out_of_plane * yhat

    # TODO Consider requiring that dot(analyzer_position - sample_position, yhat) is zero?

    sample_analyzer_center_distance = _norm(sample_analyzer_center_vector)

    analyzer_detector_center_distance = _norm(analyzer_detector_center_vector)

    # similar-triangles give the out-of-plane analyzer reflection point distance
    sa_out_of_plane = sample_analyzer_center_distance / (sample_analyzer_center_distance + analyzer_detector_center_distance) * sd_out_of_plane

    return sample_analyzer_center_vector + sa_out_of_plane * yhat


def analyzer_detector_vector(sample_position: Variable, sample_analyzer_vec: Variable, detector_position: Variable) -> Variable:
    return detector_position - (sample_position + sample_analyzer_vec)


def kf_hat(sample_analyzer_vec: Variable) -> Variable:
    return sample_analyzer_vec / _norm(sample_analyzer_vec)


def kf_wavenumber(sample_analyzer_vec: Variable, analyzer_detector_vec: Variable, tau: Variable) -> Variable:
    from scipp import sqrt, DType
    if tau.dtype == DType.vector3:
        tau = _norm(tau)

    # law of Cosines gives the scattering angle based on distances:
    l_sa = _norm(sample_analyzer_vec)
    l_ad = _norm(analyzer_detector_vec)
    l_diff = _norm(sample_analyzer_vec - analyzer_detector_vec)
    cos2theta = (l_sa * l_sa + l_ad * l_ad - l_diff * l_diff) / (2 * l_sa + l_ad)

    # law of Cosines gives the Bragg reflected wavevector magnitude
    return tau / sqrt(2 - 2 * cos2theta)


def kf_vector(kf_direction: Variable, kf_magnitude: Variable) -> Variable:
    return kf_direction * kf_magnitude


def secondary_flight_path_length(sample_analyzer_vec: Variable, analyzer_detector_vec: Variable) -> Variable:
    return _norm(sample_analyzer_vec) + _norm(analyzer_detector_vec)


def secondary_flight_time(secondary_flight_distance: Variable, kf_magnitude: Variable) -> Variable:
    from scipp.constants import hbar, neutron_mass
    velocity = kf_magnitude * hbar / neutron_mass
    return secondary_flight_distance / velocity


def sample_time(detector_time: Variable, secondary_time: Variable) -> Variable:
    return detector_time - secondary_time
