import numpy as np
import scipp as sc

from ..reflectometry.normalization import (
    reduce_from_events_to_lz,
    reduce_from_events_to_q,
    reduce_from_lz_to_q,
)


def solve_for_calibration_parameters(Io, Is):
    Iopp, Iopa, Ioap, Ioaa = Io
    Ipp, Ipa, Iap, Iaa = Is

    I0 = 2 * (Iopp * Ioaa - Iopa * Ioap) / (Iopp + Ioaa - Iopa - Ioap)
    rho = (Ioaa - Ioap) / (Iopp - Iopa)
    alp = (Ioaa - Iopa) / (Iopp - Ioap)

    Rspp_plus_Rsaa = (
        4
        * (rho * alp * Ipp + Iaa + rho * Ipa + alp * Iap)
        / ((1 + rho) * (1 + alp) * I0)
    )
    Pp = sc.sqrt(
        (Ipp + Iaa - Ipa - Iap)
        * (alp * (Ipp - Iap) - Iaa + Ipa)
        / (
            (rho * alp * Ipp + Iaa + rho * Ipa + alp * Iap)
            * (rho * (Ipp - Ipa) - Iaa + Iap)
        )
    )
    Ap = sc.sqrt(
        (Ipp + Iaa - Ipa - Iap)
        * (rho * (Ipp - Ipa) - Iaa + Iap)
        / (
            (rho * alp * Ipp + Iaa + rho * Ipa + alp * Iap)
            * (alp * (Ipp - Iap) - Iaa + Ipa)
        )
    )
    Rs = sc.sqrt(
        (alp * (Ipp - Iap) - Iaa + Ipa)
        * (rho * (Ipp - Ipa) - Iaa + Iap)
        / ((rho * alp * Ipp + Iaa + rho * Ipa + alp * Iap) * (Ipp + Iaa - Ipa - Iap))
    )

    Pa = -rho * Pp
    Aa = -alp * Ap

    Rspp_minus_Rsaa = Rs * Rspp_plus_Rsaa
    Rspp = (Rspp_plus_Rsaa + Rspp_minus_Rsaa) / 2
    Rsaa = Rspp_plus_Rsaa - Rspp

    return I0 / 4, Pp, Pa, Ap, Aa, Rspp, Rsaa


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
        * (
            Rpp * (1 + Ap) * (1 + Pp)
            + Rpa * (1 - Ap) * (1 + Pp)
            + Rap * (1 + Ap) * (1 - Pp)
            + Raa * (1 - Ap) * (1 - Pp)
        ),
        I0
        * (
            Rpp * (1 + Aa) * (1 + Pp)
            + Rpa * (1 - Aa) * (1 + Pp)
            + Rap * (1 + Aa) * (1 - Pp)
            + Raa * (1 - Aa) * (1 - Pp)
        ),
        I0
        * (
            Rpp * (1 + Ap) * (1 + Pa)
            + Rpa * (1 - Ap) * (1 + Pa)
            + Rap * (1 + Ap) * (1 - Pa)
            + Raa * (1 - Ap) * (1 - Pa)
        ),
        I0
        * (
            Rpp * (1 + Aa) * (1 + Pa)
            + Rpa * (1 - Aa) * (1 + Pa)
            + Rap * (1 + Aa) * (1 - Pa)
            + Raa * (1 - Aa) * (1 - Pa)
        ),
    )


def correction_matrix(Pp, Pa, Ap, Aa):
    return [
        [
            (1 + Pp) * (1 + Ap),
            (1 + Pp) * (1 - Ap),
            (1 - Pp) * (1 + Ap),
            (1 - Pp) * (1 - Ap),
        ],
        [
            (1 + Pp) * (1 + Aa),
            (1 + Pp) * (1 - Aa),
            (1 - Pp) * (1 + Aa),
            (1 - Pp) * (1 - Aa),
        ],
        [
            (1 + Pa) * (1 + Ap),
            (1 + Pa) * (1 - Ap),
            (1 - Pa) * (1 + Ap),
            (1 - Pa) * (1 - Ap),
        ],
        [
            (1 + Pa) * (1 + Aa),
            (1 + Pa) * (1 - Aa),
            (1 - Pa) * (1 + Aa),
            (1 - Pa) * (1 - Aa),
        ],
    ]


def compute_calibration_factors(Io, Is):
    I0, Pp, Pa, Ap, Aa, _, _ = solve_for_calibration_parameters(Io, Is)
    return I0, correction_matrix(Pp, Pa, Ap, Aa)


def linsolve(A, b):
    return np.linalg.solve(
        np.stack([[a.values for a in row] for row in A]),
        np.stack([bi.values for bi in b], axis=-1),
    )


def computer_reflectivity_calibrate_on_q(
    reference_supermirror,
    reference_polarized_supermirror,
    sample,
    qbins,
):
    reference_supermirror = [
        reduce_from_lz_to_q(i, qbins) for i in reference_supermirror
    ]
    reference_polarized_supermirror = [
        reduce_from_lz_to_q(i, qbins) for i in reference_polarized_supermirror
    ]
    sample = [reduce_from_events_to_q(i, qbins) for i in sample]
    I0, C = compute_calibration_factors(
        reference_supermirror, reference_polarized_supermirror
    )
    return [i / I0 for i in linsolve(C, sample)]


def computer_reflectivity_calibrate_on_lz(
    reference_supermirror,
    reference_polarized_supermirror,
    sample,
    wbins,
    qbins,
):
    sample = reduce_from_events_to_lz(sample, wbins)
    I0, C = compute_calibration_factors(
        reference_supermirror, reference_polarized_supermirror
    )
    sample = linsolve(C, sample)
    I0 = reduce_from_lz_to_q(I0, qbins)
    return [i / I0 for i in reduce_from_lz_to_q(sample, qbins)]
