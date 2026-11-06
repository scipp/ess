# ESTIA

## Reduction Workflows

::::{grid} 3

:::{grid-item-card} Reduction of McStas Data
:link: estia-mcstas-reduction.ipynb
:text-align: center

```{image} ../../_static/thumbnails/estia_mcstas_reduction_light.svg
:class: only-light
:width: 100%
```
```{image} ../../_static/thumbnails/estia_mcstas_reduction_dark.svg
:class: only-dark
:width: 100%
```
:::

:::{grid-item-card} Advanced reduction of McStas Data
:link: estia-advanced-mcstas-reduction.ipynb
:text-align: center

```{image} ../../_static/thumbnails/estia_advanced_mcstas_reduction_light.svg
:class: only-light
:width: 100%
```
```{image} ../../_static/thumbnails/estia_advanced_mcstas_reduction_dark.svg
:class: only-dark
:width: 100%
```
:::

::::

## Angles and alignment

ESTIA computes angles in the laboratory x-z plane, with y vertical.
The sample is aligned so that the central reflected beam hits the detector
centre. After alignment, each sample rotation about y is accompanied by twice
that rotation of the detector. Other alignment rotations stay fixed.

The sample mount angle (`RawSampleRotation`) is measured from laboratory +x
towards -z using the mount's transformed local +x axis projected onto the x-z
plane. The sample may be tilted within its mount. Supply
`SampleRotationOffset` to calibrate the angle, for example from a fit to a known
critical edge. The corrected `SampleRotation` is the angle between the central
reflected beam and the sample surface. Sample and reference runs can have
separate offsets.

The detector reference is its transformed local +z direction. It must point
from the sample to the detector centre. `divergence_angle` measures each
outgoing ray's angle from this direction, positive from laboratory +z towards
+x. The reflection angle is
`theta = divergence_angle + sample_rotation`.

```{toctree}
---
hidden:
---

estia-mcstas-reduction
estia-advanced-mcstas-reduction
simulated-spin-flip-sample
create-estia-wavelength-lookup-table
```
