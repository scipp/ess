import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_allclose

from ess.estia.calibration import (
    correction_matrix,
    linsolve,
    solve_for_calibration_parameters,
)


def generate_valid_calibration_parameters():
    I0 = np.random.random()
    Pp = np.random.random()
    Pa = -np.random.random()
    Ap = np.random.random()
    Aa = -np.random.random()
    Rspp = np.random.random()
    Rsaa = Rspp * np.random.random()
    return tuple(map(sc.scalar, (I0, Pp, Pa, Ap, Aa, Rspp, Rsaa)))


def intensity_from_parameters(I0, Pp, Pa, Ap, Aa, Rpp, Rpa, Rap, Raa):
    return (
        I0
        / 4
        * (
            Rpp * (1 + Ap) * (1 + Pp)
            + Rpa * (1 - Ap) * (1 + Pp)
            + Rap * (1 + Ap) * (1 - Pp)
            + Raa * (1 - Ap) * (1 - Pp)
        ),
        I0
        / 4
        * (
            Rpp * (1 + Aa) * (1 + Pp)
            + Rpa * (1 - Aa) * (1 + Pp)
            + Rap * (1 + Aa) * (1 - Pp)
            + Raa * (1 - Aa) * (1 - Pp)
        ),
        I0
        / 4
        * (
            Rpp * (1 + Ap) * (1 + Pa)
            + Rpa * (1 - Ap) * (1 + Pa)
            + Rap * (1 + Ap) * (1 - Pa)
            + Raa * (1 - Ap) * (1 - Pa)
        ),
        I0
        / 4
        * (
            Rpp * (1 + Aa) * (1 + Pa)
            + Rpa * (1 - Aa) * (1 + Pa)
            + Rap * (1 + Aa) * (1 - Pa)
            + Raa * (1 - Aa) * (1 - Pa)
        ),
    )


@pytest.mark.parametrize("seed", range(10))
def test_calibration_solve_recovers_input(seed):
    np.random.seed(seed)
    I0, Pp, Pa, Ap, Aa, Rspp, Rsaa = generate_valid_calibration_parameters()
    Io = intensity_from_parameters(
        I0, Pp, Pa, Ap, Aa, sc.scalar(1), sc.scalar(0), sc.scalar(0), sc.scalar(1)
    )
    Is = intensity_from_parameters(
        I0, Pp, Pa, Ap, Aa, Rspp, sc.scalar(0), sc.scalar(0), Rsaa
    )
    tuple(
        map(
            assert_allclose,
            solve_for_calibration_parameters(Io, Is),
            (I0, Pp, Pa, Ap, Aa, Rspp, Rsaa),
        )
    )


@pytest.mark.parametrize(('dims', 'shape'), [('x', (5,)), ('xy', (2, 3))])
def test_stacking_in_linsolve(dims, shape):
    A = [
        [sc.array(dims=dims, values=np.random.randn(*shape)) for _ in range(4)]
        for _ in range(4)
    ]
    x = [sc.array(dims=dims, values=np.random.randn(*shape)) for _ in range(4)]
    b = [sum(xi * ai for xi, ai in zip(x, a, strict=True)) for a in A]
    for u, v in zip(linsolve(A, b), x, strict=True):
        assert_allclose(u, v, atol=sc.scalar(1e-9))


@pytest.mark.parametrize("seed", range(5))
def test_calibration_factor_matches_intensity_from_parameters(seed):
    np.random.seed(seed)
    I0, Pp, Pa, Ap, Aa, _, _ = (
        v.value for v in generate_valid_calibration_parameters()
    )
    Rpp, Rpa, Rap, Raa = np.random.random(4)
    assert np.allclose(
        intensity_from_parameters(I0, Pp, Pa, Ap, Aa, Rpp, Rpa, Rap, Raa),
        I0
        * np.array(correction_matrix(Pp, Pa, Ap, Aa))
        @ np.array([Rpp, Rpa, Rap, Raa]),
    )
