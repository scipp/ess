# MAGIC

MAGIC — **Mag**netism Single-Crystal Diffractometer — is an ESS polarised
time-of-flight single-crystal diffractometer for studying magnetic structures.

## Reading McStas simulations

`MagicMcStasWorkflow` reads raw events from one MAGIC absorption logger and can independently construct its detector geometry.
Raw-event loading reads every row of the selected bank into memory, so large simulations require substantial RAM.

```python
from ess.magic import MagicMcStasWorkflow
from ess.magic.types import DetectorBank, Filename, RawMcStasEvents, SampleRun

workflow = MagicMcStasWorkflow()
workflow[Filename[SampleRun]] = 'simulation.h5'
events = workflow.compute(RawMcStasEvents[SampleRun])
```

Bank A (`A_abs_logger`) is the default.
To read bank B (`B_abs_logger`) from a simulation that contains it, set
`workflow[DetectorBank] = DetectorBank.detector_b` before computing.

The result is a flat Scipp DataArray with an `event` dimension:

| Field | Meaning | Unit |
| --- | --- | --- |
| Data | Original simulation weight `p` | `counts` (weighted-count convention) |
| Variances | `p**2`, assuming independent rows | `counts**2` |
| `x`, `y`, `z` | Absorption position in the logger's local frame | `m` |
| `vx`, `vy`, `vz` | Velocity in the logger's local frame | `m/s` |
| `t` | Original McStas arrival time | `s` |

Weights are not normalized to an absolute count rate.
The variance convention matches the MAGIC Mantid converter, but does not represent correlations between rows introduced by McStas splitting.
Row order and zero-weight rows are preserved.
The reader does not apply detector acceptance cuts, assign pixels, correct times, or calculate wavelength.
These operations will be added as separate providers.

## Detector geometry

Compute the selected bank's geometry without loading its event data:

```python
from ess.magic.types import EmptyDetector

detector = workflow.compute(EmptyDetector[SampleRun])
```

The result has a `pixel_id` dimension, integer pixel IDs as data and a coordinate,
and a `position` coordinate in metres in the global McStas frame.
Bank A has IDs 1–491520; bank B has IDs 491521–622592.
The geometry uses the Mantid workflow's cassette model and its default shared-plane convention:
the two counters of each cassette share the same pixel-centre positions.

Bank A uses the simulation's `A_r1`, `A_section_length`, `A_casette_gamma`, and `A_casette_omega` parameters.
Bank B uses the reference model's fixed radius, length, spacing and channel counts, and requires `B_casette_omega`.
Both banks require their absorption logger's position and rotation.
Missing parameters are not replaced with defaults.

For comparison with Mantid's sample-centred instrument definition, subtract the global sample position from these pixel positions.
Mantid fixes bank A's radius, length and spacing to 1 m, 0.53 m and 1.05 degrees; comparisons with other values require adapting the reference model.
This provider constructs geometry only; assigning the raw events to these pixels is a separate step.
