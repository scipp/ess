from ..diffraction.peaks import dspacing_peaks_from_cif
from .types import CIFIdentifierForPeakPositions, CIFPeaksMinIntensity, DHKLList


def dhkl_peaks_from_cif(
    cif: CIFIdentifierForPeakPositions, intensity_threshold: CIFPeaksMinIntensity
) -> DHKLList:
    '''Gets the list of expected peak positions from a CIF file/identifier.'''
    return dspacing_peaks_from_cif(cif, intensity_threshold).coords['dspacing']
