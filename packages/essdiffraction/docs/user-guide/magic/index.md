# MAGIC

MAGIC — **Mag**netism Single-Crystal Diffractometer — is an ESS polarised
time-of-flight single-crystal diffractometer for studying magnetic structures.

## Reading McStas simulations

`MagicMcStasWorkflow` currently reads raw events from one MAGIC absorption logger.
It loads every row of the selected bank into memory, so large simulations require substantial RAM.

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
