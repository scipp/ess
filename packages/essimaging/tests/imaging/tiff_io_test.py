# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import ess.odin.data
import scipp as sc
from scipp.testing import assert_identical
from scitiff.io import load_scitiff

from ess.imaging.io import (
    _add_to_event_time_offset_in_case_of_pulse_skipping,
    tiff_from_event_data,
)


def test_tiff_dumping_helper(tmp_path: Path):
    small_iron_image = ess.odin.data.iron_simulation_sample_small()
    output_path = tmp_path / "dump_timepix.tiff"

    tiff_from_event_data(
        small_iron_image,
        output_path,
        time_bins=20,
        pulse_stride=2,
    )
    # Test if the saved tiff file has expected output.
    loaded = load_scitiff(file_path=output_path, only_image=True)
    assert loaded.sizes['t'] == 20
    assert 'x_pixel_offset' in loaded.coords
    assert 'y_pixel_offset' in loaded.coords


def test_correct_event_time_offset() -> None:
    assert_identical(
        _add_to_event_time_offset_in_case_of_pulse_skipping(
            sc.datetimes(dims='t', values=[0, 1, 2], unit='s'),
            pulse_stride=2,
            pulse_period=sc.scalar(1, unit='s'),
        ),
        sc.array(dims='t', values=[0, 1.0, 0], unit='s'),
    )
    assert_identical(
        _add_to_event_time_offset_in_case_of_pulse_skipping(
            sc.datetimes(dims='t', values=[0, 1, 2], unit='s'),
            pulse_stride=2,
            pulse_period=sc.scalar(1, unit='s'),
            pulse_stride_offset=1,
        ),
        sc.array(dims='t', values=[1.0, 0, 1.0], unit='s'),
    )
    assert_identical(
        _add_to_event_time_offset_in_case_of_pulse_skipping(
            sc.datetimes(dims='t', values=[10, 999, 2100], unit='ms'),
            pulse_stride=2,
            pulse_period=sc.scalar(1, unit='s'),
        ),
        sc.array(dims='t', values=[0, 1.0, 0], unit='s'),
    )
    assert_identical(
        _add_to_event_time_offset_in_case_of_pulse_skipping(
            sc.datetimes(dims='t', values=[-100, 999, 2100], unit='ms'),
            pulse_stride=3,
            pulse_period=sc.scalar(1, unit='s'),
        ),
        sc.array(dims='t', values=[2, 0.0, 1], unit='s'),
    )
    assert_identical(
        _add_to_event_time_offset_in_case_of_pulse_skipping(
            sc.datetimes(dims='t', values=[-100, 999, 2100], unit='ms'),
            pulse_stride=3,
            pulse_stride_offset=2,
            pulse_period=sc.scalar(1, unit='s'),
        ),
        sc.array(dims='t', values=[1, 2.0, 0], unit='s'),
    )
