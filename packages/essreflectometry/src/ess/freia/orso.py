# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import os

from orsopy.fileio import File

from ess.reflectometry.orso import OrsoCorrectionList, OrsoReferenceFilenames
from ess.reflectometry.types import Filename, ReferenceRun


def orso_freia_corrections() -> OrsoCorrectionList:
    """Return list of corrections applied in Freia reductions."""
    return OrsoCorrectionList(
        [
            "chopper ToF correction",
            "footprint correction",
            "direct beam normalization",
        ]
    )


def orso_freia_reference_files(
    reference_filename: Filename[ReferenceRun],
) -> OrsoReferenceFilenames:
    """Collect the no-sample direct-beam reference file used by FREIA."""
    return OrsoReferenceFilenames(
        [File(file=os.path.basename(reference_filename), comment="direct beam")]
        if reference_filename
        else []
    )


providers = (orso_freia_corrections, orso_freia_reference_files)
