from ess.spectroscopy.types import *


def sample_analyzer_vector(
    sample_position: SamplePosition,
    analyzer_position: AnalyzerPosition,
    analyzer_orientation: AnalyzerOrientation,
    detector_position: DetectorPosition,
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
    from ess.spectroscopy.utils import norm
    from scipp import dot, vector

    # Scipp does not distinguish between coordinates and directions, so we need to do some extra legwork
    # to ensure we can apply the orientation transformation _and_ obtain a dimensionless direction vector
    o = vector([0, 0, 0], unit=analyzer_orientation.unit)
    y = vector(
        [0, 1, 0], unit=analyzer_orientation.unit
    )  # and y perpendicular to the scattering plane
    yhat = analyzer_orientation * y - analyzer_orientation * o
    yhat /= norm(yhat)

    sample_analyzer_center_vector = analyzer_position - sample_position

    sample_detector_vector = detector_position - sample_position
    sd_out_of_plane = dot(sample_detector_vector, yhat)
    # the sample-detector vector is the sum of sample-analyzer, analyzer-detector-center, the out-of-plane vector
    analyzer_detector_center_vector = (
        sample_detector_vector - sample_analyzer_center_vector - sd_out_of_plane * yhat
    )

    # TODO Consider requiring that dot(analyzer_position - sample_position, yhat) is zero?

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
    from scipp import atan2, dot, vector

    lab_x = vector(
        [1, 0, 0]
    )  # perpendicular to the incident beam, in the horizontal plane
    lab_z = vector([0, 0, 1])  # along the incident beam direction
    return atan2(y=dot(lab_x, vec), x=dot(lab_z, vec)).to(unit='deg')


def fixed_tau_hat_sample_analyzer_vector(
    sample_position: SamplePosition,
    analyzer_position: AnalyzerPosition,
    analyzer_orientation: AnalyzerOrientation,
    detector_position: DetectorPosition,
) -> SampleAnalyzerVector:
    """Determine the sample to analyzer-reflection-point vector per detector element as the intersection of three planes

    :parameter sample_position: scipp.DType.vector3
        The (probably singular) sample position, typically (0, 0, 0)
    :parameter analyzer_position: scipp.DType.vector3
        Any point on the central analyzer blade reflecting plane
    :parameter analyzer_orientation: scipp.DType.rotate3
        The orienting quaternion of the analyzer, used to identify the crystal tau and y-axis
    :parameter detector_position: scipp.DType.vector3
        The position of the detector element

    :Note:
    The shapes of the analyzer position and orientation should be self-consistent and will likely be 1:1.
    There is expected to be multiple detector element positions per analyzer which can be represented as
    an additional dimension compared to the analyzer shapes.

    :Warning:
    This function makes the incorrect assumption that the _direction_ of tau is known, but this breaks the
    prismatic analyzer model for BIFROST. In reality, we rely on the finite mosaic spread of the analyzer to
    scatter slightly different energies to the different detectors, so this method can not work.
    """
    from uuid import uuid4

    from ess.spectroscopy.utils import norm
    from numpy.linalg import lstsq as linalg_least_sqr
    from numpy.linalg import matrix_rank
    from numpy.linalg import solve as linalg_solve
    from scipp import any as sc_any
    from scipp import concat, cross, dot, scalar, vector, vectors

    # Scipp does not distinguish between coordinates and directions, so we need to do some extra legwork
    # to ensure we can apply the orientation transformation _and_ obtain a dimensionless direction vector
    o = vector([0, 0, 0], unit=analyzer_orientation.unit)
    x = vector(
        [1, 0, 0], unit=analyzer_orientation.unit
    )  # McStas defines x parallel to Bragg plane tau
    y = vector(
        [0, 1, 0], unit=analyzer_orientation.unit
    )  # and y perpendicular to the scattering plane
    # z is in the Bragg plane and in the scattering plane

    tau, yhat = [
        v / norm(v)
        for v in [
            (analyzer_orientation * ei - analyzer_orientation * o) for ei in (x, y)
        ]
    ]

    v_d = detector_position - sample_position
    n_v_d = norm(v_d)
    if sc_any(n_v_d == scalar(0.0, unit=sample_position.unit)):
        bad = n_v_d == scalar(0.0, unit=sample_position.unit)
        n_v_d.values[bad.values] = 1.0
        print(f"Some detector(s) are located at the detector?")
    v_d /= n_v_d

    # collect the normal directions for the three planes
    n_1, n_2, n_3 = cross(v_d, tau), tau, v_d  # each (N_det, 3), or (3,)
    # and collect the right-hand side of the three plane equations
    b_1 = dot(sample_position, n_1).to(
        unit=sample_position.unit
    )  # (N_sample, N_det) or (N_det,)
    b_2 = dot(analyzer_position, n_2).to(unit=sample_position.unit)  # (N_analyzer,)
    b_3 = dot(0.5 * sample_position + 0.5 * detector_position, n_3).to(
        unit=sample_position.unit
    )

    if (
        n_1.ndim > 0
        or n_2.ndim > 0
        or n_3.ndim > 0
        or b_1.ndim > 0
        or b_2.ndim > 0
        or b_3.ndim > 0
    ):
        blob = 0 * (n_1 + n_2 + n_3)  # broadcast to a common shape
        n_1, n_2, n_3 = [x + blob for x in (n_1, n_2, n_3)]
        blob = 0 * (b_1 + b_2 + b_3)  # broadcast to a common shape
        b_1, b_2, b_3 = [x + blob for x in (b_1, b_2, b_3)]

    new_dim = str(uuid4())
    a = (
        concat([n_1, n_2, n_3], dim=new_dim)
        .transpose(dims=n_1.dims + (new_dim,))
        .values
    )  # (..., 3, 3)
    b = (
        concat([b_1, b_2, b_3], dim=new_dim)
        .transpose(dims=n_1.dims + (new_dim,))
        .values
    )  # (..., 3)

    # as of numpy 2.0, b must be (..., 3, 1) for broadcasting to work
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    b = b[..., :, None]
    # and we need to strip off this dimension after solving
    x = linalg_solve(a, b).squeeze(-1)

    # if any of the n_1, n_2, or n_3 are zero (or, equivalently, a linear relationship exists between any two),
    # numpy.linalg.solve will not have worked, and probably returned garbage.
    if any((mr := matrix_rank(a)) < 3):
        from numpy import ndenumerate

        for i, m in ndenumerate(mr):
            if m < 3:
                x[i] = linalg_least_sqr(a[i], b[i][:, 0])[0]

    analyzer_point = vectors(values=x, dims=n_1.dims, unit=sample_position.unit)
    return analyzer_point - sample_position


def analyzer_detector_vector(
    sample_position: SamplePosition,
    sample_analyzer_vec: SampleAnalyzerVector,
    detector_position: DetectorPosition,
) -> AnalyzerDetectorVector:
    return detector_position - (sample_position + sample_analyzer_vec)


def kf_hat(sample_analyzer_vec: SampleAnalyzerVector) -> SampleAnalyzerDirection:
    from ess.spectroscopy.utils import norm

    return sample_analyzer_vec / norm(sample_analyzer_vec)


def reciprocal_lattice_spacing(tau_vector: ReciprocalLatticeVectorAbsolute):
    from ess.spectroscopy.utils import norm

    return norm(tau_vector)


def final_wavenumber(
    sample_analyzer_vec: SampleAnalyzerVector,
    analyzer_detector_vec: AnalyzerDetectorVector,
    tau: ReciprocalLatticeSpacing,
) -> FinalWavenumber:
    from ess.spectroscopy.utils import norm
    from scipp import sqrt

    # law of Cosines gives the scattering angle based on distances:
    l_sa = norm(sample_analyzer_vec)
    l_ad = norm(analyzer_detector_vec)
    l_diff = norm(sample_analyzer_vec + analyzer_detector_vec)
    # 2 theta is measured from the direction S-A, so the internal angle is (pi - 2 theta)
    # and the normal law of Cosines is modified accordingly to be -cos(2 theta) instead of cos(pi - 2 theta)
    cos2theta = (l_diff * l_diff - l_sa * l_sa - l_ad * l_ad) / (2 * l_sa * l_ad)

    # law of Cosines gives the Bragg reflected wavevector magnitude
    return tau / sqrt(2 - 2 * cos2theta)


def final_energy(kf: FinalWavenumber) -> FinalEnergy:
    from scipp.constants import hbar, neutron_mass

    return ((hbar * hbar / 2 / neutron_mass) * kf * kf).to(unit='meV')


def final_wavevector(
    kf_direction: SampleAnalyzerDirection, kf_magnitude: FinalWavenumber
) -> FinalWavevector:
    return kf_direction * kf_magnitude


def secondary_flight_path_length(
    sample_analyzer_vec: SampleAnalyzerVector,
    analyzer_detector_vec: AnalyzerDetectorVector,
) -> SampleDetectorPathLength:
    from ess.spectroscopy.utils import norm

    return norm(sample_analyzer_vec) + norm(analyzer_detector_vec)


def secondary_flight_time(
    secondary_flight_distance: SampleDetectorPathLength, kf_magnitude: FinalWavenumber
) -> SampleDetectorFlightTime:
    from scipp.constants import hbar, neutron_mass

    velocity = kf_magnitude * hbar / neutron_mass
    return secondary_flight_distance / velocity


def sample_frame_time(
    detector_time: DetectorFrameTime, secondary_time: SampleDetectorFlightTime
) -> SampleFrameTime:
    return detector_time - secondary_time


providers = [
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
]
