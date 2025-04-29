from dataclasses import dataclass

import numpy as np
import scipp as sc

from ..reflectometry.normalization import (
    reduce_from_events_to_lz,
    reduce_to_q,
)
from ..reflectometry.types import CoordTransformationGraph, QBins, WavelengthBins
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


def _kronecker_product(A, B):
    return [
        [A[ia][ja] * B[ib][jb] for ja in range(2) for jb in range(2)]
        for ia in range(2)
        for ib in range(2)
    ]


@dataclass
class PolarizationCalibrationParameters:
    I0: sc.DataArray
    '''Reference intensity.'''
    Pp: sc.DataArray
    '''Effective polarization of polarizer when polarizer flipper is off.'''
    Pa: sc.DataArray
    '''Effective polarization of polarizer when polarizer flipper is on.'''
    Ap: sc.DataArray
    '''Effective polarization of analyzer when analyzer flipper if on.'''
    Aa: sc.DataArray
    '''Effective polarization of analyzer when analyzer flipper is off.'''
    Rspp: sc.DataArray
    '''Magnetic supermirror reflectivity for neutrons with
    spin parallel to instrument polarization.'''
    Rsaa: sc.DataArray
    '''Magnetic supermirror reflectivity for neutrons with
    spin anti-parallel to instrument polarization.'''

    @property
    def polarization_matrix(self):
        """
        The linear relationship :math:`\\mathbf{C}`
        such that :math:`\\mathbf{I} = I0 \\mathbf{C} \\mathbf{R}`.

        Returns
        ----------
        :
            The polarization matrix.
        """
        return _kronecker_product(
            [
                [(1 + self.Pp) / 2, (1 - self.Pp) / 2],
                [(1 + self.Pa) / 2, (1 - self.Pa) / 2],
            ],
            [
                [(1 + self.Ap) / 2, (1 - self.Ap) / 2],
                [(1 + self.Aa) / 2, (1 - self.Aa) / 2],
            ],
        )

    @classmethod
    def from_reference_measurements(cls, Io, Is):
        """
        Solves for the calibration parameters given the reference
        measurements.

        See https://doi.org/10.1016/S0921-4526(00)00823-1.

        Parameters
        ------------
        Io:
            Intensity from measurements with perfect
            non-magnetic supermirror.
        Is:
            Intensity from measurements with
            magnetic supermirror.

        Returns
        ----------
        :
            Polarization calibration parameters
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
            / (
                (rho * alp * Ipp + Iaa + rho * Ipa + alp * Iap)
                * (Ipp + Iaa - Ipa - Iap)
            )
        )

        Pa = -rho * Pp
        Aa = -alp * Ap

        Rspp_minus_Rsaa = Rs * Rspp_plus_Rsaa
        Rspp = (Rspp_plus_Rsaa + Rspp_minus_Rsaa) / 2
        Rsaa = Rspp_plus_Rsaa - Rspp

        return cls(I0, Pp, Pa, Ap, Aa, Rspp, Rsaa)

    def polarization_correction(self, Is):
        '''Corrects the intensities for imperfections of polarizing components'''
        return _linsolve(self.polarization_matrix, Is)

    def correct_intensities_and_normalize_by_reference(self, Is):
        '''Corrects the intensities for imperfections of polarizing components
        and normalizes by reference intensity to obtain reflectivity.'''
        return [i / self.I0 for i in self.polarization_correction(Is)]


def _linsolve_numpy(A, b):
    x = np.linalg.solve(
        np.stack([np.stack(row, -1) for row in A], -2),
        np.stack(b, -1)[..., None],
    )[..., 0]
    return np.moveaxis(x, -1, 0)


def _linsolve(A, b):
    '''
    Solves :math:`\\mathbf{b}=\\mathbf{A}\\mathbf{x}`
    at each point of the ``DataArrays`` in ``A`` and ``b``.

    Parameters
    ------------
    A:
        4x4 matrix containing `scipp.DataArray` or `scipp.Variable``.
    b:
        4-vector containing `scipp.DataArray` or `scipp.Variable``.

    Returns
    ----------
    :
        4-vector containing `scipp.DataArray` or `scipp.Variable``,
        representing the solution to the equation.
    '''
    x = _linsolve_numpy(
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
    return PolarizationCalibrationParameters.from_reference_measurements(
        [reduce_to_q(i, qbins) for i in reference_supermirror],
        [reduce_to_q(i, qbins) for i in reference_polarized_supermirror],
    ).correct_intensities_and_normalize_by_reference(
        [reduce_to_q(i, qbins).hist() for i in sample]
    )


def compute_reflectivity_calibrate_on_lz(
    reference_supermirror,
    reference_polarized_supermirror,
    sample,
    wbins,
    qbins,
    graph,
):
    """Applied the polarization correction on the wavelength-z grid
    then reduces to Q to apply the normalization."""
    sample = [reduce_from_events_to_lz(s, wbins).hist() for s in sample]
    cal = PolarizationCalibrationParameters.from_reference_measurements(
        reference_supermirror, reference_polarized_supermirror
    )
    for i, s in enumerate(cal.polarization_correction(sample)):
        sample[i].data = s
        sample[i] = sample[i].transform_coords(
            ("Q",),
            graph,
            rename_dims=False,
            keep_intermediate=False,
            keep_aliases=False,
        )
        sample[i].coords['Q'] = sc.midpoints(sample[i].coords['Q'], 'wavelength')

    masks = [sc.isnan(cal.I0.data) | sc.isnan(s.data) for s in sample]
    sample = [
        reduce_to_q(s.assign_masks(isnan=m), qbins)
        for s, m in zip(sample, masks, strict=True)
    ]
    return [
        s / reduce_to_q(cal.I0.assign_masks(isnan=m), qbins).data
        for s, m in zip(sample, masks, strict=True)
    ]


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


def reflectivity_provider_calibrate_on_lz(
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
    graph: CoordTransformationGraph,
) -> PolarizedReflectivityOverQ:
    return compute_reflectivity_calibrate_on_lz(
        [i000, i001, i010, i011],
        [im00, im01, im10, im11],
        [is00, is01, is10, is11],
        wbins,
        qbins,
        graph,
    )


providers = (reflectivity_provider,)
