import numpy as np
import scipp as sc

from ..reflectometry.normalization import (
    reduce_from_events_to_lz,
    reduce_from_events_to_q,
    reduce_from_lz_to_q,
)
from ..reflectometry.types import QBins, WavelengthBins
from .types import (
    Intensity,
    MagneticReference,
    MagneticSample,
    NonMagneticReference,
    OffOff,
    OffOn,
    OnOff,
    OnOn,
    PolarizedReflectivityOverQ,
)


def solve_for_calibration_parameters(Io, Is):
    """
    Solves for the calibration parameters given the reference
    measurements.

    See https://doi.org/10.1016/S0921-4526(00)00823-1.
    """
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


def correction_matrix(Pp, Pa, Ap, Aa):
    """
    Defines the linear relationship between measured intensity
    and reflectivity.
    """
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


def calibration_factors_from_reference_measurements(Io, Is):
    """
    Computes the polarization instrument parameters from
    the calibration measurements on the non-magnetic reference
    and the calibration measurements on the magnetic reference.
    """
    I0, Pp, Pa, Ap, Aa, _, _ = solve_for_calibration_parameters(Io, Is)
    return I0, correction_matrix(Pp, Pa, Ap, Aa)


def _linsolve(A, b):
    x = np.linalg.solve(
        np.stack([np.stack(row, -1) for row in A], -2),
        np.stack(b, -1)[..., None],
    )[..., 0]
    return np.moveaxis(x, -1, 0)


def linsolve(A, b):
    x = _linsolve(
        [[a.values for a in row] for row in A],
        [bi.values for bi in b],
    )
    return [sc.array(dims=b[0].dims, values=xi) for xi in x]


def compute_reflectivity_calibrate_on_q(
    reference_supermirror,
    reference_polarized_supermirror,
    sample,
    qbins,
):
    """Reduces reference and sample to Q before applying
    the polarization correction and normalization."""
    reference_supermirror = [
        reduce_from_lz_to_q(i, qbins) for i in reference_supermirror
    ]
    reference_polarized_supermirror = [
        reduce_from_lz_to_q(i, qbins) for i in reference_polarized_supermirror
    ]
    I0, C = calibration_factors_from_reference_measurements(
        reference_supermirror, reference_polarized_supermirror
    )
    sample = [reduce_from_events_to_q(i, qbins) for i in sample]
    sample = linsolve(C, sample)
    return [i / I0 for i in sample]


def compute_reflectivity_calibrate_on_lz(
    reference_supermirror,
    reference_polarized_supermirror,
    sample,
    wbins,
    qbins,
):
    """Applied the polarization correction on the wavelength-z grid
    then reduces to Q to apply the normalization."""
    sample = [reduce_from_events_to_lz(s, wbins) for s in sample]
    I0, C = calibration_factors_from_reference_measurements(
        reference_supermirror, reference_polarized_supermirror
    )
    sample = linsolve(C, sample)
    sample = [reduce_from_lz_to_q(s, qbins) for s in sample]
    I0 = reduce_from_lz_to_q(I0, qbins)
    return [i / I0 for i in sample]


def reflectivity_provider(
    i000: Intensity[NonMagneticReference, OffOff],
    i001: Intensity[NonMagneticReference, OffOn],
    i010: Intensity[NonMagneticReference, OnOff],
    i011: Intensity[NonMagneticReference, OnOn],
    im00: Intensity[MagneticReference, OffOff],
    im01: Intensity[MagneticReference, OffOn],
    im10: Intensity[MagneticReference, OnOff],
    im11: Intensity[MagneticReference, OnOn],
    is00: Intensity[MagneticSample, OffOff],
    is01: Intensity[MagneticSample, OffOn],
    is10: Intensity[MagneticSample, OnOff],
    is11: Intensity[MagneticSample, OnOn],
    wbins: WavelengthBins,
    qbins: QBins,
) -> PolarizedReflectivityOverQ:
    return compute_reflectivity_calibrate_on_q(
        [i000, i001, i010, i011],
        [im00, im01, im10, im11],
        [is00, is01, is10, is11],
        qbins,
    )


providers = (reflectivity_provider,)
