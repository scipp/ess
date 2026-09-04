# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from pathlib import Path

import ess.odin.data
from scitiff.io import load_scitiff

from ess.imaging.io import tiff_from_event_data


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
