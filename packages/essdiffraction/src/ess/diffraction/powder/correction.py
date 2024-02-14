# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc


def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    """
    Return a scipp.DataArray containing calibration metadata as coordinates.

    Parameters
    ----------
    into:
        Base data and metadata for the returned object.
    calibration:
        Calibration parameters.

    Returns
    -------
    :
        Copy of `into` with additional coordinates and masks
        from `calibration`.

    See Also
    --------
    ess.diffraction.load_calibration
    """
    dim = calibration.dim
    if not sc.identical(into.coords[dim], calibration.coords[dim]):
        raise ValueError(
            f'Coordinate {dim} of calibration and target dataset do not agree.'
        )
    out = into.copy(deep=False)
    for name in ('difa', 'difc', 'tzero'):
        if name in out.coords:
            raise ValueError(
                f"Cannot add calibration parameter '{name}' to data, "
                "there already is metadata with the same name."
            )
        out.coords[name] = calibration[name].data
    if 'calibration' in out.masks:
        raise ValueError(
            "Cannot add calibration mask 'calibration' tp data, "
            "there already is a mask with the same name."
        )
    out.masks['calibration'] = calibration['mask'].data
    return out


def lorentz_factor(data: sc.DataArray) -> sc.DataArray:
    """Compute the ToF powder diffraction Lorentz factor.

    This function uses this definition:

    .. math::

        L = d^4 \\sin\\theta

    where :math:`d` is the d-spacing, :math:`\\theta` is half the scattering angle
    (note the definitions in
    https://scipp.github.io/scippneutron/user-guide/coordinate-transformations.html).

    The Lorentz factor as defined here is suitable for correcting time-of-flight data
    expressed in wavelength or d-spacing.
    It follows the definition used bty GSAS-II, see page 140 of
    https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf

    Parameters
    ----------
    data:
        Input data with coordinates ``two_theta`` and ``dspacing``.

    Returns
    -------
    :
        Powder Lorentz factor.
    """

    def f(dspacing: sc.Variable, two_theta: sc.Variable) -> sc.Variable:
        return dspacing**4 * sc.sin(two_theta / 2)

    aux = data.transform_coords(
        'lorentz_factor', {'lorentz_factor': f}, rename_dims=False, quiet=True
    )
    if aux.bins is not None:
        return aux.bins.coords['lorentz_factor']
    return sc.DataArray(
        aux.coords['lorentz_factor'],
        coords={
            'dspacing': data.coords['dspacing'],
            'two_theta': data.coords['two_theta'],
        },
    )
