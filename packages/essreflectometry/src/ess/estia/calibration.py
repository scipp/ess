from dataclasses import dataclass
from typing import Self

import scipp as sc


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

    @classmethod
    def from_reference_measurements(
        cls: type[Self],
        Io: tuple[sc.DataArray, sc.DataArray, sc.DataArray, sc.DataArray],
        Is: tuple[sc.DataArray, sc.DataArray, sc.DataArray, sc.DataArray],
    ) -> Self:
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

        num_base = rho * alp * Ipp + Iaa + rho * Ipa + alp * Iap
        den_base = Ipp + Iaa - Ipa - Iap
        term_rho = rho * (Ipp - Ipa) - Iaa + Iap
        term_alp = alp * (Ipp - Iap) - Iaa + Ipa

        Rspp_plus_Rsaa = 4 * num_base / ((1 + rho) * (1 + alp) * I0)

        Pp = sc.sqrt(den_base * term_alp / (num_base * term_rho))
        Ap = sc.sqrt(den_base * term_rho / (num_base * term_alp))
        Rs = sc.sqrt(term_alp * term_rho / (num_base * den_base))

        Pa = -rho * Pp
        Aa = -alp * Ap

        Rspp_minus_Rsaa = Rs * Rspp_plus_Rsaa
        Rspp = (Rspp_plus_Rsaa + Rspp_minus_Rsaa) / 2
        Rsaa = Rspp_plus_Rsaa - Rspp

        return cls(I0, Pp, Pa, Ap, Aa, Rspp, Rsaa)


providers = ()
