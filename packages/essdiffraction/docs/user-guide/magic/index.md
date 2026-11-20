# MAGIC

MAGIC — **Mag**netism Single-Crystal Diffractometer — is an ESS polarised
time-of-flight single-crystal diffractometer for studying magnetic structures.

## Reading McStas simulations

`MagicMcStasWorkflow` reads weighted events from one MAGIC absorption logger and groups them by detector pixel.
It loads every row of the selected bank into memory, so large simulations require substantial RAM.

```python
from ess.magic import MagicMcStasWorkflow
from ess.magic.types import DetectorBank, Filename, RawDetector, SampleRun

workflow = MagicMcStasWorkflow()
workflow[Filename[SampleRun]] = 'simulation.h5'
detector = workflow.compute(RawDetector[SampleRun])
```

Bank A (`A_abs_logger`) is the default.
To read bank B (`B_abs_logger`) from a simulation that contains it, set
`workflow[DetectorBank] = DetectorBank.detector_b` before computing.

The result uses the shared `RawDetector` type, with a `pixel_id` dimension,
pixel positions in the common simulation frame, and weighted events in each pixel:

| Field | Meaning | Unit |
| --- | --- | --- |
| Event data | Original simulation weight `p` | `counts` (weighted-count convention) |
| `position` | Detector pixel centre | `m` |
| `t` | Original McStas arrival time | `s` |
| `event_time_offset` | Arrival time relative to the most recent pulse | `s` |
| `event_time_zero` | Pulse timestamp using a synthetic Unix-epoch origin | `datetime64[ns]` |

Weights are not normalized to an absolute count rate.
Zero-weight events and empty pixels are retained; events outside the detector are excluded.
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

geometry = workflow.compute(EmptyDetector[SampleRun])
```

The result has a `pixel_id` dimension, integer pixel IDs as data and a coordinate,
and a `position` coordinate in metres in the common simulation frame.
Bank A has IDs 1–491520; bank B has IDs 491521–622592.
