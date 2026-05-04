# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.reflectometry.orso import OrsoCorrectionList


def orso_freia_corrections() -> OrsoCorrectionList:
    """Return list of corrections applied in Freia reductions."""
    return OrsoCorrectionList(
        [
            "chopper ToF correction",
            "footprint correction",
        ]
    )


providers = (orso_freia_corrections,)
