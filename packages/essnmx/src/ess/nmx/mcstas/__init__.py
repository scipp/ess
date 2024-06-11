# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib


def McStasWorkflow():
    import sciline as sl

    from ess.nmx.reduction import bin_time_of_arrival

    from .load import providers as loader_providers
    from .xml import read_mcstas_geometry_xml

    return sl.Pipeline(
        (*loader_providers, read_mcstas_geometry_xml, bin_time_of_arrival)
    )
