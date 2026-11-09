# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from pathlib import Path

import scipp as sc
from ess.powder.masking import apply_masks, read_pixel_masks
from ess.powder.types import PixelMaskFilenames


def test_apply_masks_masks_pixels_listed_in_file(tmp_path: Path) -> None:
    path = tmp_path / 'broken.h5'
    masked = sc.array(dims=['id'], values=[2], unit=None, dtype='int32')
    sc.io.save_hdf5(masked, path)
    data = sc.DataArray(
        sc.ones(sizes={'detector_number': 3}),
        coords={
            'detector_number': sc.array(
                dims=['detector_number'], values=[1, 2, 3], unit=None
            )
        },
    )

    result = apply_masks(
        data=data,
        pixel_masks=read_pixel_masks(PixelMaskFilenames([path])),
        tof_mask_func=None,
        wavelength_mask_func=None,
        two_theta_mask_func=None,
    )

    assert result.masks['broken.h5'].values.tolist() == [False, True, False]
