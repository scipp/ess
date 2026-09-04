# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import io
from pathlib import Path

import scipp as sc
import scippnexus as snx
import scitiff


def _add_to_event_time_offset_in_case_of_pulse_skipping(
    event_time_zero: sc.Variable,
    pulse_stride: int,
    pulse_period: sc.Variable,
) -> sc.Variable:
    _pulse_period = pulse_period.to(unit=event_time_zero.unit)
    etz = event_time_zero - sc.datetime(0, unit=event_time_zero.unit)
    # The offset is used to place some etz value in the center of a binning
    # where the bins have constant width pulse_period.
    # That way small deviations in etz will not move the etz to the next
    # or previous bin and each subsequent pulse will have a different index.
    offset = _pulse_period / 2 - etz.nanmin() % _pulse_period
    index = ((etz + offset) // _pulse_period) % pulse_stride
    return index * pulse_period


def tiff_from_event_data(
    nexus_file_name: str | Path | io.BytesIO,
    output_path: str | Path | io.BytesIO,
    *,
    time_bins: int | sc.Variable,
    pulse_stride: int,
    detector_group_path: str = "/entry/instrument/event_mode_detectors/timepix3",
    event_data_field_path: str = "timepix3_events",
) -> None:
    '''
    Write a tiff image file representing the data from the nexus file.

    Parameters
    ------------
    nexus_file_name:
        The file name of the nexus file to write to tiff.
    output_path:
        Where to write the tiff file.
    time_bins:
        The number of time slices the image should have.
    pulse_stride:
        The pulse stride that was used when doing the measurement.
    detector_group_path:
        Path to the event data group in the nexus file.
    event_data_field_path:
        Path to the event data group in the nexus file.
    '''
    with snx.File(nexus_file_name) as f:
        data = f[detector_group_path][()][event_data_field_path]

    data.bins.coords['event_time_offset'] += (
        _add_to_event_time_offset_in_case_of_pulse_skipping(
            data.bins.coords['event_time_zero'],
            pulse_stride=pulse_stride,
            pulse_period=sc.scalar(1 / 14, unit="s").to(
                unit=data.bins.coords['event_time_offset'].unit
            ),
        )
    )
    # In case dimension names fall back to default ones.
    ydim_name = 'dim_0' if 'dim_0' in data.dims else 'y_pixel_offset'
    xdim_name = 'dim_1' if 'dim_1' in data.dims else 'x_pixel_offset'
    image = data.hist(event_time_offset=time_bins).rename_dims(
        {'event_time_offset': 't', ydim_name: 'y', xdim_name: 'x'}
    )
    image = image.drop_coords([c for c in image.coords if image.coords[c].ndim > 1])
    scitiff.save_scitiff(image, output_path)
