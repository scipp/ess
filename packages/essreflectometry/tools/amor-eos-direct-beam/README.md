# Amor direct-beam reduction in scipp, validated against eos 3.3.4

`amor_scipp.py` is a standalone reduction of Amor (PSI/SINQ) event data written with
scipp, scippnexus and the detector pixel table of `ess.amor.geometry`. It follows the
direct-beam (`-nm d`) algorithm of the PSI reduction program eos 3.3.4
(https://gitea.psi.ch/sinq-reflectometry/eos, commit `68e84a8`) step by step, and takes the
same command-line options, so that the two can be compared number by number.

It is a validation tool, not part of the `ess.amor` workflow.

`amor-direct-beam-reduction.ipynb` is a tutorial notebook in the style of the Amor user-guide
notebooks. It walks through the same reduction step by step: loading a run, the detector
images, the wavelength maps of direct beam and sample, the angle and Q of every cell, the
projection to R(Q), three angles in one go, the comparison with eos, and the effect of the
convention differences. It is stored without outputs; the example raw files are not
distributed, so point `raw_path` at your own Amor files. `build_notebook.py` regenerates it.

The script exposes the steps as functions: `make_config`, `load_run`, `pixel_table`,
`detector_image`, `direct_beam_map`, `sample_map`, `lz_grid`, `project_to_q`, `reduce`.

```bash
python amor_scipp.py -f 6310-6312 -n 6328 -o out -rp raw \
       -r 0.05 -q 0.005 0.27 -l 3 12 -m -0.055 -T -0.055 0.06 -y 18 48
```

## Agreement with eos

Tested on one dataset of an October 2026 beamtime: a silicon block, bare and with a lipid
bilayer, in D2O and H2O, three incident angles each (12 reduced curves, 1 to 5 runs per
curve, direct-beam normalisation, 5 % Q bins). Versions: scipp 26.9.0, scippnexus 26.1.1,
essreflectometry 26.11.0.

With `--convention eos` (default), on the identical Q grid:

| quantity | largest relative difference over all valid bins of all 12 curves |
|---|---|
| R | 9.6e-8 |
| sigma R | 9.6e-8 |
| sigma Q | 1.5e-13 |
| set of valid bins | identical |

The residual is one constant factor per curve: eos keeps the per-pulse monitor in single
precision.

## What the algorithm does

1. Event time of flight: shift by the chopper trigger phase offset, fold into one frame,
   subtract the chopper-opening delay seen at each detector angle; wavelength from the
   flight path chopper to pixel.
2. Filters: pulses whose proton current is below threshold (with their events), detector
   strips outside the y range, wavelength outside the range, event Q outside the requested
   range.
3. Monitor: proton current valid at each pulse times the pulse length, summed over accepted
   pulses. Pulse times come from the beam-monitor group when the file has no chopper
   trigger events.
4. Histogram sample and direct beam on (wavelength, detector z). The direct-beam map is
   mirrored in z.
5. Per (wavelength, z) cell: angle = nu - mu + pixel angle + gravity term, Q = 4 pi sin/lambda.
   For multi-run curves the first run's angles define this grid.
6. R(Q) = sum of sample counts / sum of direct-beam counts over the unmasked cells in each
   Q bin, times the monitor ratio.

## Why `ess.amor.AmorWorkflow` was not used directly

* File layout: the 2026 files keep events in `entry1/detector/data`, angles in
  `entry1/measurement_configuration`, scalar chopper fields, no chopper trigger events.
* Normalisation: the workflow assumes a supermirror reference with footprint correction;
  a direct beam needs neither, and needs the z mirror.
* Monitor: eos weights and filters per pulse by proton current.

## Convention differences between eos and `ess.amor`

`--convention ess` switches to the choices in `ess.amor`; `--ess-parts` selects them
individually. Effect on one curve (bare substrate, D2O, middle angle) relative to eos:

| convention | max rel. change of R in a bin | rms | max change / sigma R |
|---|---|---|---|
| path inside the detector `(nWires-1-wire)*dX` instead of `wire*dX` | 12 % | 3.8 % | 1.35 |
| h/m_n from `scipp.constants` instead of the eos constant | 0.14 % | 0.06 % | 0.04 |
| gravity `asin(sin(theta) + g L lambda^2 m_n^2 / 2h^2)` instead of `theta + atan(3.07e-10 L lambda^2)` | 0.84 % | 0.22 % | 0.25 |

The detector-depth convention is the one that matters. It is marked as an open question
in `ess/amor/geometry.py` (`# TODO: check with Jochen if this is correct`). It changes each
event's wavelength by up to 0.65 %, more than the 0.5 % wavelength bin of the map.

## Not implemented

Supermirror / over-illumination normalisation, time slicing, polarisation splitting, files
with chopper trigger events, full ORSO header on output.
