# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .types import IofQ, NormalizedIofQ, Reference, Sample


def normalize_by_supermirror(
    sample: IofQ[Sample],
    supermirror: IofQ[Reference],
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
    normalized = sample / supermirror
    normalized.masks['no_reference_neutrons'] = (supermirror == sc.scalar(0)).data
    return NormalizedIofQ(normalized)


providers = (normalize_by_supermirror,)
