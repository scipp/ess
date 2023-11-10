# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

# from ..reflectometry import orso
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
    # TODO
    # try:
    #    normalized.attrs['orso'] = sample.attrs['orso']
    #    normalized.attrs['orso'].value.reduction.corrections = list(
    #        set(
    #            sample.attrs['orso'].value.reduction.corrections
    #            + supermirror.attrs['orso'].value.reduction.corrections
    #        )
    #    )
    #    normalized.attrs[
    #        'orso'
    #    ].value.data_source.measurement.reference = supermirror.attrs[
    #        'orso'
    #    ].value.data_source.measurement.data_files
    # except KeyError:
    #    orso.not_found_warning()
    return NormalizedIofQ(normalized)


providers = [normalize_by_supermirror]
