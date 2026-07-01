# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Default parameter specs for SANS workflows.
"""

from __future__ import annotations

import sciline
import scipp as sc
from pydantic import BaseModel, Field

from ess.reduce.parameter import ParameterRegistry, ParameterSpec
from ess.reduce.parameter_models import (
    LengthUnit,
    QEdges,
    QUnit,
    WavelengthEdges,
)

from ..sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    BeamCenter,
    CorrectedDetector,
    CorrectForGravity,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    IntensityQ,
    IntensityQxQy,
    NeXusDetectorName,
    NeXusMonitorName,
    Numerator,
    PixelMaskFilename,
    PixelShapePath,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    SampleRun,
    TransformationPath,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
    WavelengthMonitor,
)


class QxEdges(QEdges):
    """Model for Qx edges."""

    start: float = Field(default=-0.5, description="Start of the Qx edges.")
    stop: float = Field(default=0.5, description="Stop of the Qx edges.")

    def get_edges(self) -> sc.Variable:
        return sc.linspace(
            dim='Qx',
            start=self.start,
            stop=self.stop,
            num=self.num_bins + 1,
            unit=self.unit.value,
        )


class QyEdges(QEdges):
    """Model for Qy edges."""

    start: float = Field(default=-0.5, description="Start of the Qy edges.")
    stop: float = Field(default=0.5, description="Stop of the Qy edges.")

    def get_edges(self) -> sc.Variable:
        return sc.linspace(
            dim='Qy',
            start=self.start,
            stop=self.stop,
            num=self.num_bins + 1,
            unit=self.unit.value,
        )


class BeamCenterXY(BaseModel):
    """Beam center position in detector coordinates."""

    x: float = Field(default=0.0, description="Beam center x coordinate.")
    y: float = Field(default=0.0, description="Beam center y coordinate.")
    unit: LengthUnit = Field(default=LengthUnit.METER, description="Beam center unit.")

    def get_vector(self) -> sc.Variable:
        return sc.vector([self.x, self.y, 0.0], unit=self.unit.value)


def _edges(model):
    return model.get_edges()


def _beam_center(model: BeamCenterXY) -> sc.Variable:
    return model.get_vector()


def _direct_beam_filename(
    workflow: sciline.Pipeline, filename: str | None
) -> sciline.Pipeline:
    workflow = workflow.copy()
    if filename:
        workflow[DirectBeamFilename] = filename
    else:
        workflow[DirectBeam] = None
    return workflow


parameters = ParameterRegistry()


parameters[Filename[SampleRun]] = ParameterSpec(
    model=tuple[str, ...],
    category='Files',
    title='Sample Runs',
    description='Comma-separated NeXus file paths for the sample runs.',
)
parameters[Filename[BackgroundRun]] = ParameterSpec(
    model=tuple[str, ...],
    category='Files',
    title='Background Runs',
    description='Comma-separated NeXus file paths for the background runs.',
)
parameters[Filename[TransmissionRun[SampleRun]]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Sample Transmission',
    description='NeXus file path for the sample transmission run.',
    default=None,
)
parameters[Filename[TransmissionRun[BackgroundRun]]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Background Transmission',
    description='NeXus file path for the background transmission run.',
    default=None,
)
parameters[Filename[EmptyBeamRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Empty Beam',
    description='NeXus file path for the empty-beam run.',
    default=None,
)
parameters[PixelMaskFilename] = ParameterSpec(
    model=tuple[str, ...],
    category='Files',
    title='Pixel Masks',
    description='Comma-separated paths to detector pixel mask files.',
    default=(),
)
parameters[DirectBeamFilename] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Direct Beam',
    description='Direct-beam file path; leave empty to skip direct-beam correction.',
    default=None,
    apply=_direct_beam_filename,
)

parameters[NeXusDetectorName] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Detector',
    description='Name of the detector group in the NeXus files.',
)
parameters[NeXusMonitorName[Incident]] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Incident Monitor',
    description='Name of the incident monitor group in the NeXus files.',
    default='',
)
parameters[NeXusMonitorName[Transmission]] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Transmission Monitor',
    description='Name of the transmission monitor group in the NeXus files.',
    default='',
)
parameters[TransformationPath] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Transform Path',
    description='NeXus path containing detector transformation information.',
    default='',
)
parameters[PixelShapePath] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Pixel Shape Path',
    description='NeXus path containing detector pixel-shape information.',
    default='',
)

parameters[WavelengthBins] = ParameterSpec(
    model=WavelengthEdges,
    category='Binning',
    title='Wavelength Edges',
    description='Wavelength bin edges used for monitor and detector histograms.',
    default=WavelengthEdges(start=2.0, stop=12.0, num_bins=300),
    transform=_edges,
    use_workflow_default=False,
)
parameters[QBins] = ParameterSpec(
    model=QEdges,
    category='Binning',
    title='Q Edges',
    description='Q bin edges used for one-dimensional I(Q) outputs.',
    default=QEdges(start=0.1, stop=0.3, num_bins=100, unit=QUnit.INVERSE_ANGSTROM),
    transform=_edges,
    use_workflow_default=False,
)
parameters[QxBins] = ParameterSpec(
    model=QxEdges,
    category='Binning',
    title='Qx Edges',
    description='Qx bin edges used for two-dimensional I(Qx, Qy) outputs.',
    default=QxEdges(),
    transform=_edges,
    use_workflow_default=False,
)
parameters[QyBins] = ParameterSpec(
    model=QyEdges,
    category='Binning',
    title='Qy Edges',
    description='Qy bin edges used for two-dimensional I(Qx, Qy) outputs.',
    default=QyEdges(),
    transform=_edges,
    use_workflow_default=False,
)

parameters[CorrectForGravity] = ParameterSpec(
    model=bool,
    category='Reduction',
    title='Correct Gravity',
    description='Apply gravity correction to neutron trajectories before reduction.',
    default=False,
    transform=CorrectForGravity,
)
parameters[ReturnEvents] = ParameterSpec(
    model=bool,
    category='Reduction',
    title='Return Events',
    description='Keep event data in outputs where the workflow supports it.',
    default=False,
    transform=ReturnEvents,
)
parameters[UncertaintyBroadcastMode] = ParameterSpec(
    model=UncertaintyBroadcastMode,
    category='Reduction',
    title='Uncertainty Broadcast',
    description='How uncertainties are broadcast when reduced data are combined.',
    default=UncertaintyBroadcastMode.upper_bound,
)
parameters[BeamCenter] = ParameterSpec(
    model=BeamCenterXY,
    category='Reduction',
    title='Beam Center',
    description='Detector-plane beam center used for Q conversion.',
    default=BeamCenterXY(),
    transform=_beam_center,
    use_workflow_default=False,
)


typical_outputs = (
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    IntensityQ[SampleRun],
    IntensityQxQy[SampleRun],
    IntensityQ[BackgroundRun],
    IntensityQxQy[BackgroundRun],
    CorrectedDetector[BackgroundRun, Numerator],
    CorrectedDetector[SampleRun, Numerator],
    WavelengthMonitor[SampleRun, Incident],
    WavelengthMonitor[SampleRun, Transmission],
    WavelengthMonitor[BackgroundRun, Incident],
    WavelengthMonitor[BackgroundRun, Transmission],
)
