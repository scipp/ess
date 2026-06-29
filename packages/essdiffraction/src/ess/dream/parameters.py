# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Default parameter specs for DREAM workflows."""

from __future__ import annotations

from math import pi

from ess.powder.masking import with_pixel_mask_filenames
from ess.powder.types import (
    CalibrationFilename,
    DspacingBins,
    EmptyCanRun,
    Filename,
    IntensityDspacing,
    IntensityDspacingTwoTheta,
    MonitorFilename,
    NeXusDetectorName,
    PixelMaskFilename,
    ReducedTofCIF,
    SampleRun,
    TwoThetaBins,
    UncertaintyBroadcastMode,
    VanadiumRun,
)

from ess.reduce.parameter import ParameterRegistry, ParameterSpec
from ess.reduce.parameter_models import AngleUnit, DspacingEdges, TwoTheta

from .beamline import InstrumentConfiguration


def _edges(model):
    return model.get_edges()


parameters = ParameterRegistry()

parameters[Filename[SampleRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Sample Run',
    description='NeXus file path for the sample run.',
    default=None,
)
parameters[Filename[VanadiumRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Vanadium Run',
    description='NeXus file path for the vanadium normalization run.',
    default=None,
)
parameters[Filename[EmptyCanRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Empty Can Run',
    description='NeXus file path for the empty-can background run.',
    default=None,
)
parameters[CalibrationFilename] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Calibration',
    description='Path to the calibration file used for detector calibration.',
    default=None,
)
parameters[MonitorFilename[SampleRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Sample Monitor',
    description='NeXus file path for the sample monitor data.',
    default=None,
)
parameters[MonitorFilename[VanadiumRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Vanadium Monitor',
    description='NeXus file path for the vanadium monitor data.',
    default=None,
)
parameters[PixelMaskFilename] = ParameterSpec(
    model=tuple[str, ...],
    category='Files',
    title='Pixel Masks',
    description='Comma-separated paths to detector pixel mask files.',
    default=(),
    apply=with_pixel_mask_filenames,
)

parameters[NeXusDetectorName] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Detector',
    description='Name of the detector group in the NeXus files.',
    default='mantle',
)

parameters[DspacingBins] = ParameterSpec(
    model=DspacingEdges,
    category='Binning',
    title='D-spacing Edges',
    description='D-spacing bin edges used for diffraction intensity outputs.',
    default=DspacingEdges(start=0.0, stop=2.0, num_bins=200),
    transform=_edges,
    use_workflow_default=False,
)
parameters[TwoThetaBins] = ParameterSpec(
    model=TwoTheta,
    category='Binning',
    title='Two-Theta Edges',
    description='Two-theta bin edges used for two-dimensional diffraction outputs.',
    default=TwoTheta(start=0.0, stop=pi, num_bins=180, unit=AngleUnit.RADIAN),
    transform=_edges,
    use_workflow_default=False,
)

parameters[InstrumentConfiguration] = ParameterSpec(
    model=InstrumentConfiguration,
    category='Reduction',
    title='Instrument Configuration',
    description='DREAM chopper system configuration used in the measurement.',
    default=InstrumentConfiguration.high_flux_BC215,
)
parameters[UncertaintyBroadcastMode] = ParameterSpec(
    model=UncertaintyBroadcastMode,
    category='Reduction',
    title='Uncertainty Broadcast',
    description='How uncertainties are treated when uncertain quantities are combined.',
    default=UncertaintyBroadcastMode.upper_bound,
)


typical_outputs = (
    IntensityDspacing[SampleRun],
    IntensityDspacingTwoTheta[SampleRun],
    ReducedTofCIF,
)
