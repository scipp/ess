import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

from ess.estia.calibration import PolarizationCalibrationParameters


def _kronecker_product(A, B):
    return [
        [A[ia][ja] * B[ib][jb] for ja in range(2) for jb in range(2)]
        for ia in range(2)
        for ib in range(2)
    ]


def _matvec(A, b):
    return [sum(A[i][j] * b[j] for j in range(len(b))) for i in range(len(A))]


def generate_valid_calibration_parameters():
    I0 = np.random.random()
    Pp = np.random.random()
    Pa = -np.random.random()
    Ap = np.random.random()
    Aa = -np.random.random()
    Rspp = np.random.random()
    Rsaa = Rspp * np.random.random()
    return tuple(map(sc.scalar, (I0, Pp, Pa, Ap, Aa, Rspp, Rsaa)))


def polarization_matrix_from_beamline_parameters(Pp, Pa, Ap, Aa):
    return _kronecker_product(
        [
            [(1 + Pp) / 2, (1 - Pp) / 2],
            [(1 + Pa) / 2, (1 - Pa) / 2],
        ],
        [
            [(1 + Ap) / 2, (1 - Ap) / 2],
            [(1 + Aa) / 2, (1 - Aa) / 2],
        ],
    )


@pytest.mark.parametrize("seed", range(10))
def test_calibration_solve_recovers_input(seed):
    np.random.seed(seed)
    I0, Pp, Pa, Ap, Aa, Rspp, Rsaa = generate_valid_calibration_parameters()
    polmat = polarization_matrix_from_beamline_parameters(Pp, Pa, Ap, Aa)
    supermirror_reflectivities = [
        sc.scalar(1),
        sc.scalar(0),
        sc.scalar(0),
        sc.scalar(1),
    ]
    magnetic_supermirror_reflectivities = [Rspp, sc.scalar(0), sc.scalar(0), Rsaa]
    Io = [I0 * v for v in _matvec(polmat, supermirror_reflectivities)]
    Is = [I0 * v for v in _matvec(polmat, magnetic_supermirror_reflectivities)]
    cal = PolarizationCalibrationParameters.from_reference_measurements(Io, Is)
    tuple(
        map(
            assert_allclose,
            (cal.I0, cal.Pp, cal.Pa, cal.Ap, cal.Aa, cal.Rspp, cal.Rsaa),
            (I0, Pp, Pa, Ap, Aa, Rspp, Rsaa),
        )
    )
