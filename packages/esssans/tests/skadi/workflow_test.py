# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import scipp as sc
import scippnexus as snx
from ess.sans.types import Position, RawDetector, SampleRun, SolidAngle
from ess.skadi import SkadiWorkflow


def test_workflow_computes_solid_angle_from_calibrated_detector() -> None:
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
    workflow = SkadiWorkflow()
    workflow[RawDetector[SampleRun]] = detector
    workflow[Position[snx.NXsample, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')

    solid_angle = workflow.compute(SolidAngle[SampleRun])

    assert solid_angle.sizes == detector.sizes
    assert solid_angle.unit == 'dimensionless'
    assert sc.all(
        sc.isfinite(solid_angle.data) & (solid_angle.data > sc.scalar(0))
    ).value
