# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import warnings

import scipp as sc
import scippnexus as snx

from .configurations import InputConfig


def read_goniometer_values(
    file: snx.Group,
    input_config: InputConfig,
) -> sc.Variable:
    """Read goniometer values from MANDI file.

    Parameters
    ----------
    file:
        Mandi file object to read goniometer values from.
    input_config:
        Mandi reduction input configuration.
        Configuration object should contain
        paths to :math:`\\chi` (chi), :math:`\\phi` (phi),
        and :math:`\\omega` (omega) values.
        Each path is expected to have NXlog group.
        See :obj:`ess.mandi.configurations.InputConfig` for more details.

    Returns
    -------
    :
        :math:`\\chi`, :math:`\\phi`, :math:`\\omega` values as a vector.

    """
    chi_path = input_config.gonio_path_chi
    phi_path = input_config.gonio_path_phi
    omega_path = input_config.gonio_path_omega
    chi, phi, omega = file[chi_path][()], file[phi_path][()], file[omega_path][()]

    # validate the goniometer values
    def _validate(da) -> None:
        if da.ndim > 1:
            raise ValueError("Goniometer chi value must scalar or 1D.")
        elif da.ndim != 0 and da.sizes[da.dim] > 1:
            warnings.warn(
                "More than 1 values found for gonio meter value. "
                "Average value will be used.",
                category=UserWarning,
                stacklevel=2,
            )

    for val in (chi, phi, omega):
        _validate(val)

    def _retrieve_value(da) -> sc.Variable:
        average_key = 'average_value'
        if isinstance(da, sc.Variable):
            var = da
        elif average_key in da.coords:
            var = da.coords[average_key]
        else:
            var = da.data
        if var.unit is None:
            warnings.warn(
                "No unit specified for goniometer value. "
                "Inserting expected unit, 'deg'...",
                category=UserWarning,
                stacklevel=2,
            )
            var.unit = 'deg'

        if var.ndim > 0:
            var = var.mean()
        return var

    values = [_retrieve_value(val) for val in (chi, phi, omega)]
    if len(unit_set := {val.unit for val in values}) != 1:
        warnings.warn(
            "Units for goniometer values don't match. Using 'deg' by default...",
            category=UserWarning,
            stacklevel=2,
        )
    unit = next(iter(unit_set))

    crystal_rotation = sc.vector([val.value for val in values], unit=unit)
    if crystal_rotation.unit is None:
        warnings.warn(
            "No unit specified for crystal rotation. Inserting expected unit, 'deg'",
            category=UserWarning,
            stacklevel=2,
        )
        crystal_rotation.unit = 'deg'
    return crystal_rotation
