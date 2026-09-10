# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx
from ess.sans.types import Position, RawDetector, SampleRun, SolidAngle
from ess.skadi import SkadiWorkflow
from scipp.testing import assert_allclose, assert_identical


def test_workflow_computes_solid_angle_preserving_pixel_masks() -> None:
    detector = sc.DataArray(
        sc.ones(sizes={'detector_number': 2}),
        coords={
            'position': sc.vectors(
                dims=['detector_number'],
                values=[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
                unit='m',
            ),
            'pixel_size': sc.vectors(
                dims=['detector_number'],
                values=[[0.02, 0.03, 0.001], [0.02, 0.03, 0.001]],
                unit='m',
            ),
            'detector_normal': sc.vectors(
                dims=['detector_number'],
                values=[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
                unit='dimensionless',
            ),
        },
    )
    detector = sc.broadcast(
        detector, sizes={'detector_number': 2, 'wavelength': 3}
    ).copy()
    pixel_mask = sc.array(dims=['detector_number'], values=[False, True])
    detector.masks['pixel_mask'] = pixel_mask
    detector.masks['wavelength_mask'] = sc.array(
        dims=['wavelength'], values=[False, True, False]
    )
    detector.coords['wavelength'] = sc.arange('wavelength', 4, unit='angstrom')
    workflow = SkadiWorkflow()
    workflow[RawDetector[SampleRun]] = detector
    workflow[Position[snx.NXsample, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')

    solid_angle = workflow.compute(SolidAngle[SampleRun])

    assert_allclose(
        solid_angle.data,
        sc.array(
            dims=['detector_number'],
            values=[0.00015, 0.0012 / 5**1.5],
            unit='dimensionless',
        ),
    )
    assert_identical(solid_angle.masks['pixel_mask'], pixel_mask)
    assert 'wavelength_mask' not in solid_angle.masks
    assert 'wavelength' not in solid_angle.coords
