import scipp as sc
from scippneutron.conversion.tof import dspacing_from_tof
from scipy.signal import find_peaks, medfilt

from .types import DetectorData, RunType, StreakClusteredData


def cluster_events_by_streak(da: DetectorData[RunType]) -> StreakClusteredData[RunType]:
    if isinstance(da, sc.DataGroup):
        return sc.DataGroup({k: cluster_events_by_streak(v) for k, v in da.items()})
    da = da.copy(deep=False)

    # TODO: approximate t0 should depend on chopper information
    approximate_t0 = sc.scalar(6e-3, unit='s')

    da.coords['coarse_d'] = dspacing_from_tof(
        tof=da.coords['event_time_offset'] - approximate_t0,
        Ltotal=da.coords['L0'],
        two_theta=da.coords['two_theta'],
    ).to(unit='angstrom')

    h = da.hist(coarse_d=1000)
    i_peaks, _ = find_peaks(
        h.data.values, height=medfilt(h.values, kernel_size=99), distance=3
    )
    i_valleys, _ = find_peaks(
        h.data.values.max() - h.data.values, distance=3, height=h.data.values.max() / 2
    )

    valleys = h.coords['coarse_d'][i_valleys]
    peaks = sc.array(
        dims=['coarse_d'],
        values=h.coords['coarse_d'].values[i_peaks],
        unit=h.coords['coarse_d'].unit,
    )

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
    b = da.bin(coarse_d=filtered_valleys).assign_masks(
        no_peak=has_peak != sc.scalar(1, unit=None)
    )
    b = b.drop_coords(('coarse_d',))
    b = b.bins.drop_coords(('coarse_d',))
    b = b.rename_dims(coarse_d='streak')
    return b


providers = (cluster_events_by_streak,)
