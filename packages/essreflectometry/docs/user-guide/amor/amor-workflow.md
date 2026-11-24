# Amor data reduction

{func}`ess.amor.AmorWorkflow` reduces the files that Amor writes since 2026.
The reference measurement that the sample is normalized by is either a reflection from a supermirror or a measurement of the direct beam.
The keyword argument `normalized_by` selects which:

| `normalized_by` | `Filename[ReferenceRun]` is | Treatment of the reference |
|---|---|---|
| `'supermirror'` (default) | a reflection from a supermirror | footprint correction, division by the reflectivity of the supermirror |
| `'direct_beam'` | the unrestricted beam without a sample | mirrored along the scattering direction of the detector |
| `'direct_beam_with_aperture'` | the direct beam restricted to the sample footprint at the reference angle | footprint correction at the reference angle, then mirroring |

Files recorded until 2025 have a different layout and are reduced with {func}`ess.amor.AmorWorkflow2025`, see the other pages of this section.

```{note}
There is no public example file in the current layout yet.
The code on this page is therefore not executed when the documentation is built.
```

## Normalization by the direct beam

```python
import scipp as sc

from ess import amor
from ess.reflectometry.types import *

workflow = amor.AmorWorkflow(normalized_by='direct_beam')

workflow[Filename[SampleRun]] = 'amor2026n000101.hdf'
workflow[Filename[ReferenceRun]] = 'amor2026n000100.hdf'

# Width of the sample along the beam, for the footprint correction.
workflow[SampleSize[SampleRun]] = sc.scalar(50.0, unit='mm')

# Wavelength range. The reference is histogrammed in these bins.
workflow[WavelengthBins] = sc.geomspace('wavelength', 3.0, 12.0, 280, unit='angstrom')
# Detector region: strips across the beam, rows along the scattering direction.
workflow[YIndexLimits] = sc.scalar(18), sc.scalar(48)
workflow[ZIndexLimits] = sc.scalar(0), sc.scalar(447)
# Part of the divergent beam that is used, relative to its center.
workflow[BeamDivergenceLimits] = (
    sc.scalar(-0.75, unit='deg'),
    sc.scalar(0.75, unit='deg'),
)
workflow[QBins] = sc.geomspace('Q', 0.01, 0.035, 126, unit='1/angstrom')

reflectivity = workflow.compute(ReflectivityOverQ)
reflectivity.hist().plot(norm='log')
```

The reflectivity is on an absolute scale.
Both measurements are divided by the proton charge on the target of SINQ, so they do not need to be of the same length.

With `normalized_by='direct_beam'`, the reference contains the full beam, so only the sample receives a footprint correction.
For a direct beam measured through an aperture matching the sample footprint, use:

```python
workflow = amor.AmorWorkflow(normalized_by='direct_beam_with_aperture')
```

Set the filenames and other parameters as in the example above.
This mode applies the footprint correction to both measurements, each at its own incidence angles.
The reference uses the transmitted-ray geometry and the sample uses the reflected-ray geometry.
The corrections cancel when the accepted fractions of the beam match; measurements at different angles generally need different corrections.
`SampleRotation[ReferenceRun]` must describe the orientation whose footprint the aperture represents; by default it is loaded from the reference file, including `SampleRotationOffset[ReferenceRun]`.
`SampleSize[ReferenceRun]` defaults to the sample size of the sample run and can be overridden if the aperture represents a different size.
Keep `SampleSize` and `BeamSize` at their physical values; the sample size also determines the resolution in $Q$.

## Normalization by a supermirror

```python
workflow = amor.AmorWorkflow()  # normalized_by='supermirror'

workflow[Filename[SampleRun]] = 'sample.hdf'
workflow[Filename[ReferenceRun]] = 'supermirror.hdf'
workflow[SampleSize[SampleRun]] = sc.scalar(10.0, unit='mm')
workflow[SampleSize[ReferenceRun]] = sc.scalar(10.0, unit='mm')
```

The remaining parameters are set as above.
The properties of the supermirror are set with `MValue`, `CriticalEdge` and `Alpha` from {mod}`ess.reflectometry.supermirror`.

## Adding runs

{func}`ess.reflectometry.workflow.with_filenames` adds the events of several runs.
The proton charges of the runs are added too, so the result is the total number of neutrons divided by the total charge.

```python
from ess.reflectometry.workflow import with_filenames

workflow = with_filenames(
    workflow,
    SampleRun,
    ['amor2026n000102.hdf', 'amor2026n000103.hdf', 'amor2026n000104.hdf'],
)
```

The angles of the first run are used to assign $Q$ to the reference.

## What the workflow does

1. **Loading.**
   The events are read from `entry1/detector/data` and binned by blade, wire and strip of the detector.
   The sample angle $\mu$ and the detector angle $\nu$ are read from `entry1/measurement_configuration`, the distance, speed and separation of the choppers from the chopper group of the instrument.
2. **Wavelength.**
   The chopper lets neutrons pass every half revolution, but the events are recorded relative to a trigger that occurs once per revolution.
   After correcting the trigger phase and the pixel-dependent opening time, arrival times are folded into the first pulse period, see {func}`ess.amor.conversions.wavelength`.
   `WavelengthBins` selects and histograms wavelengths without changing their assignment.
   The wavelength range must fit within one pulse period at every detector pixel: approximately 0–13.9 Å at 450 rpm with the default geometry.
   A range outside that interval raises an error.
   This assumes the instrument suppresses frame overlap; selecting a wavelength range cannot distinguish slow neutrons from earlier pulses.
3. **Angle and momentum transfer.**
   The reflection angle of each event follows from $\mu$, $\nu$ and the position of the pixel, with a correction for the drop of the neutron under gravity.
4. **Masks.**
   Events outside `YIndexLimits`, `ZIndexLimits`, `BeamDivergenceLimits` and the range of `WavelengthBins` are masked.
   Direct-beam divergence is expressed in the reflected beam's convention so the selection is applied to reference events before mirroring.
5. **Proton charge.**
   The proton current at the start of every chopper revolution is multiplied by the duration of the revolution.
   Revolutions where the current is below `ProtonCurrentThreshold` are discarded, together with their events.
   Revolutions without events count towards the charge.
6. **Reference.**
   The reference is histogrammed over detector row and wavelength.
   For a supermirror it is divided by the reflectivity of the supermirror.
   The direct beam is mirrored instead: a reflection reverses the angular profile of the beam, so the intensity that the direct beam shows in one detector row is the incident intensity for the row on the opposite side of the beam center.
7. **Reflectivity.**
   The reference is assigned the $Q$ that each detector row and wavelength has in the sample measurement.
   The sample events and the reference are histogrammed in `QBins` and divided.

## Parameters with default values

| Parameter | Default | Meaning |
|---|---|---|
| `ChopperPhaseOffset[RunType]` | -5° | Phase between the chopper trigger and the opening of the chopper |
| `ProtonCurrentThreshold` | 50 µA | Proton current below which a chopper revolution is discarded |
| `BeamDivergenceLimits` | ±0.75° | Part of the divergent beam that is used |
| `BeamSize[RunType]` | 2 mm | Full width at half maximum of the beam, for the footprint correction |
| `SampleRotationOffset[RunType]` | 0° | Correction added to the sample angle $\mu$ |
| `GravityToggle` | `True` | Correct the reflection angle for gravity |
| `DetectorSpatialResolution` | 2.5 mm | Resolution of the detector, for the resolution in $Q$ |

## Relation to the Amor reduction software

With `normalized_by='direct_beam'` the workflow uses the selection of chopper revolutions by proton current and the mirroring of the direct beam from the Amor reduction software `eos`.
The integrated proton charge is identical to the monitor value of `eos`.

The following choices differ:

- `eos` histograms the sample over detector row and wavelength and assigns $Q$ to the cells of the histogram.
  The workflow assigns $Q$ to each event of the sample.
- The path length inside the detector, the value of $h / m_n$ and the form of the gravity correction are those of {mod}`ess.amor.geometry` and {mod}`ess.amor.conversions`.
- This workflow folds corrected times into the first pulse period; `eos` 3.3.4 uses a fixed 2.5 Å cutoff when merging frames in its reflectivity workflow.
- This workflow masks direct-beam divergence per event, using transmitted-ray angles.
- The direct-beam mode in `eos` does not apply a footprint correction.
- The uncertainties are propagated from the counts of the sample measurement.
