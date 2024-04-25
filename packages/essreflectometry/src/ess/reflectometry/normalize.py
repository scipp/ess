# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .types import (
    CorrectionMatrix,
    FootprintCorrectedData,
    NormalizationFactor,
    NormalizedIofQ,
    QBins,
    Sample,
    WBins,
)


def normalization_factor(
    da: FootprintCorrectedData[Sample],
    corr: CorrectionMatrix,
    qbins: QBins,
    wbins: WBins,
) -> NormalizationFactor:
    '''The correction matrix gives us the expected intensity at each
    (z_index, wavelength) bin assuming the reflectivity is one.
    To normalize the sample measurement we need to integrate the total
    expected intensity in every Q-bin.
    Note that Q refers to the 'sample-Q', different from the 'reference-Q'.

    The 'sample-Q' is computed taking the mean of the sample measurement Q
    value in every (z_index, wavelength) bin.
    One complication however is that some bins have zero intensity from the
    sample measurement, so we are unable to assign a 'sample-Q' value to those bins.
    Therefore we estimate the intensity in the missing bins by fitting the
    'sample-q' as a function of z_index and wavelength.

    Steps:
        Approximate 'sample-q' in every (z_index, wavelength) bin
        Fit 'sample-q'.
        Compute 'sample-q' in all bins using the fit.
        Return the reference intensity with the 'sample-q' as a coordinate.

    '''
    sample_q = (
        sc.DataArray(
            data=da.coords['Q'],
            coords=dict(
                wavelength=da.coords['wavelength'],
                z_index=da.coords['z_index'],
            ),
            masks=da.masks,
        )
        # TODO: Can this be done more efficiently?
        .group('z_index')
        .bin(wavelength=wbins)
        .bins.mean()
        .data
    )

    def Q_of_z_wavelength(z_index, wavelength, a, b, c):
        z = sc.scalar(14 * 32) - sc.array(dims=z_index.dims, values=z_index.values)
        return (a + b * z + c * z**2) / wavelength

    p, _ = sc.curve_fit(
        ['z_index', 'wavelength'],
        Q_of_z_wavelength,
        sc.DataArray(
            data=sample_q,
            coords=dict(
                wavelength=corr.coords['wavelength'],
                z_index=corr.coords['z_index'],
            ),
            masks=dict(corr.masks, _sample_q_isnan=sc.isnan(sample_q)),
        ),
    )
    return sc.DataArray(
        data=corr.data,
        coords=dict(
            Q=Q_of_z_wavelength(
                corr.coords['z_index'],
                corr.coords['wavelength'],
                p['a'].data.value,
                p['b'].data.value,
                p['c'].data.value,
            ),
        ),
        masks=corr.masks,
    )


def normalize_by_supermirror(
    da: FootprintCorrectedData[Sample],
    n: NormalizationFactor,
    qbins: QBins,
) -> NormalizedIofQ:
    """
    Normalize the sample measurement by the (ideally calibrated) supermirror.

    Parameters
    ----------
    sample:
        Sample measurement with coords of 'Q' and 'detector_id'.
    supermirror:
        Supermirror measurement with coords of 'Q' and 'detector_id', ideally
        calibrated.

    Returns
    -------
    :
        normalized sample.
    """
    return NormalizedIofQ(da.bin(Q=qbins) / sc.values(n.flatten(to='Q').hist(Q=qbins)))


providers = (normalize_by_supermirror, normalization_factor)
