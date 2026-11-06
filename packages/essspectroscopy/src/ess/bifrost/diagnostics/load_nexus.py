# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from pathlib import Path
from typing import Any

import scipp as sc

from ess.reduce.diagnostics import load_nexus

load_monitors = load_nexus.load_monitors


def load_detectors(
    filename: str | Path,
    *,
    detectors: list[str] | str | None = None,
) -> sc.DataGroup[Any]:
    """Load detector data from a NeXus file.

    See :func:`ess.reduce.diagnostics.load_nexus.load_detectors`.
    """
    detectors = load_nexus.load_detectors(filename, detectors=detectors)
    return detectors.apply(lambda det: det.rename(dim_0='tube', dim_1='length'))
