# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""Input/output for DREAM."""

from .nexus import fold_nexus_detectors, load_nexus

__all__ = ["fold_nexus_detectors", "load_nexus"]
