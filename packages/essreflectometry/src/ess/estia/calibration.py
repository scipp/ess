from dataclasses import dataclass

import scipp as sc


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


providers = ()
