# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterable

import sciline

from ..nexus import GenericNeXusWorkflow
from ..nexus.types import AnyRun, FrameMonitor0
from . import WavelengthLutMode, lut, to_wavelength


def GenericUnwrapWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key],
    monitor_types: Iterable[sciline.typing.Key],
    wavelength_from: WavelengthLutMode = "file",
) -> sciline.Pipeline:
    """
    Generic workflow for computing the neutron wavelength for detector and monitor
    data.

    This workflow builds on the ``GenericNeXusWorkflow`` and computes wavelength
    from a lookup table that is created from the chopper settings, detector Ltotal and
    the neutron time-of-arrival.

    It is possible to limit which run types and monitor types
    are supported by the returned workflow.
    This is useful to reduce the size of the workflow and make it easier to inspect.
    Make sure to add *all* required run types and monitor types when using this feature.

    Attention
    ---------
    Filtering by run type and monitor type does not work with nested type vars.
    E.g., if you have a type like ``Outer[Inner[RunType]]``, this type and its
    provider will be removed.

    Parameters
    ----------
    run_types:
        List of run types to include in the workflow.
        Constrains the possible values of :class:`ess.reduce.nexus.types.RunType`.
    monitor_types:
        List of monitor types to include in the workflow.
        Constrains the possible values of :class:`ess.reduce.nexus.types.MonitorType`
        and :class:`ess.reduce.nexus.types.Component`.
    wavelength_from:
        Mode for creating the wavelength lookup table. Possible values are
        'analytical', 'simulation', and 'file'. See
        https://scipp.github.io/ess/reduce/user-guide/unwrap/lut-building-methods.html

    Returns
    -------
    :
        The workflow.
    """
    wf = GenericNeXusWorkflow(run_types=run_types, monitor_types=monitor_types)

    for provider in (
        *to_wavelength.providers(),
        *lut.providers(wavelength_from=wavelength_from),
    ):
        wf.insert(provider)
    for key, value in lut.default_parameters(wavelength_from=wavelength_from).items():
        wf[key] = value

    return wf


def LookupTableWorkflow(
    *,
    use_simulation: bool = True,
    run_types: Iterable[sciline.typing.Key] | None = None,
    monitor_types: Iterable[sciline.typing.Key] | None = None,
) -> sciline.Pipeline:
    """
    Alias for :func:`GenericUnwrapWorkflow` with default parameters set for generating
    a wavelength lookup table using a tof simulation or analytical calculations.

    This is deprecated and will be removed in a future release. Use
    :func:`GenericUnwrapWorkflow` instead with the desired parameters.

    Parameters
    ----------
    use_simulation:
        Whether to use the "simulation" or "analytical" mode for generating the lookup
        table. See :func:`GenericUnwrapWorkflow` for details.
    run_types:
        List of run types to include in the workflow.
        Constrains the possible values of :class:`ess.reduce.nexus.types.RunType`.
    monitor_types:
        List of monitor types to include in the workflow.
        Constrains the possible values of :class:`ess.reduce.nexus.types.MonitorType`
        and :class:`ess.reduce.nexus.types.Component`.
    """
    if run_types is None:
        run_types = [AnyRun]
    if monitor_types is None:
        monitor_types = [FrameMonitor0]

    return GenericUnwrapWorkflow(
        run_types=run_types,
        monitor_types=monitor_types,
        wavelength_from="simulation" if use_simulation else "analytical",
    )
