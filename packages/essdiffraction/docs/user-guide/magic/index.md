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
| `event_time_offset` | Arrival time relative to the most recent pulse | `s` |
| `event_time_zero` | Pulse timestamp using a synthetic Unix-epoch origin | `datetime64[ns]` |

Weights are not normalized to an absolute count rate.
The variance convention matches the MAGIC Mantid converter, but does not represent correlations between rows introduced by McStas splitting.
Row order and zero-weight rows are preserved.
The reader does not apply detector acceptance cuts, assign pixels, apply emission-time corrections, or calculate wavelength.
These operations will be added as separate providers.

The default pulse period is 1/14 s. Arrival times are folded into that period,
with pulse timestamps rounded to the nearest nanosecond; `t` is retained for diagnostics.
No emission-time correction is applied. To override the pulse period:

```python
import scipp as sc
from ess.reduce.unwrap import PulsePeriod

workflow[PulsePeriod] = sc.scalar(1 / 14, unit='s')
```

## Detector geometry

Compute the selected bank's geometry without loading its event data:

```python
from ess.magic.types import EmptyDetector

detector = workflow.compute(EmptyDetector[SampleRun])
```

The result has a `pixel_id` dimension, integer pixel IDs as data and a coordinate,
and a `position` coordinate in metres in the common simulation frame.
Bank A has IDs 1–491520; bank B has IDs 491521–622592.
Bank A uses the simulation's `A_r1`, `A_section_length`, `A_casette_gamma`, and `A_casette_omega` parameters.
Bank B uses the reference model's fixed radius, length, spacing and channel counts, and requires `B_casette_omega`.
Both banks require their absorption logger's position and rotation.
Missing parameters are not replaced with defaults.
