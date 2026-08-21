import scipp as sc
from ess.powder.types import ElasticCoordTransformGraph, RunType
from scipy.signal import find_peaks, medfilt

from .types import RawDetector, StreakClusteredData


def cluster_events_by_streak(
    da: RawDetector[RunType],
    graph: ElasticCoordTransformGraph[RunType],
) -> StreakClusteredData[RunType]:
    da = da.transform_coords(
        ['dspacing', 'tof', 'Ltotal', 'two_theta'],
        graph=graph,
        keep_intermediate=False,
    )
    da.bins.coords['coarse_d'] = da.bins.coords.pop('dspacing').to(unit='angstrom')

    # We need to keep these coordinates after binning,
    # adding them to the binned data coords achieves this.
    for coord in ('two_theta', 'Ltotal'):
        da.bins.coords[coord] = sc.bins_like(da, da.coords[coord])

    h = da.bins.concat().hist(coarse_d=1000)
    i_peaks, _ = find_peaks(
        h.data.values, height=medfilt(h.values, kernel_size=99), distance=3
    )
    i_valleys, _ = find_peaks(
        h.data.values.max() - h.data.values, distance=3, height=h.data.values.max() / 2
    )

    valleys = h.coords['coarse_d'][i_valleys]
    peaks = h.coords['coarse_d'][i_peaks]

    has_peak = peaks.bin(coarse_d=valleys).bins.size().data.to(dtype='bool')
    filtered_valleys = valleys[
        sc.concat(
            [
                has_peak[0],
                has_peak[:-1] | has_peak[1:],
                has_peak[-1],
            ],
            dim=has_peak.dim,
        )
    ]
    has_peak = peaks.bin(coarse_d=filtered_valleys).bins.size().data
    b = (
        da.bins.concat()
        .bin(coarse_d=filtered_valleys)
        .assign_masks(no_peak=has_peak != sc.scalar(1, unit=None))
    )
    b = b.drop_coords(('coarse_d',))
    b = b.bins.drop_coords(('coarse_d',))
    b = b.rename_dims(coarse_d='streak')
    return b
