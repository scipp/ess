# Handoff: Q resolution for esssans (scipp/ess#763)

Status: design done, no implementation yet.
Design and derivation: scipp/ess#763 (read it first; this document does not repeat the math).
Earlier attempt: scipp/esssans#185 (per pixel and wavelength, no combination rule; types are outdated, do not port directly).
Original requirement: scipp/ess#360.

## Decisions made

- Weight per-element resolution terms by the normalization term `N_i` (the existing denominator), not by detected events.
  Reason: `I_bin = Σ C / Σ N`, so the bin's resolution function is exactly the N-weighted mixture.
  Event weighting was proposed by instrument scientists; the toy model shows differences of a few percent only, so this is about correctness, noise and cost, not large numbers.
- σ per Q bin is the second moment of the mixture (`mean(σ_i²) + var(Q_i)`), not the mean of σ as in Mantid Q1D.
- Everything is accumulated as sums (`S1 = Σ N Q`, `S2 = Σ N (σ² + Q²)`, with `S0` = denominator), so banks, runs and live-data chunks merge by addition.
- Wavelength spread from the source comes from the essreduce lookup-table variance, not from an ISIS moderator file.
- Standard output is `Qdev` via variances on the `Q` coordinate. The mixture approximation (see issue) is optional and comes last.

## Suggested order of work

1. Bank merging (separate PR).
   - `ess/sans/workflow.py`: `with_banks` currently maps over `NeXusDetectorName` without reducing, and the `parameter_mappers[NeXusDetectorName]` line is commented out with a TODO.
   - Reduce at `NormalizedQ[RunType, IofQPart]` for both parts, using `merge_contributions`, the same pattern `_set_runs` uses for multiple runs.
   - Check that per-bank `DetectorBankSizes` and `DimsToKeep` still produce identical output dims across banks.
   - Update the `with_banks` docstring (the "different Q-resolution" argument).
   - Existing tests: `tests/loki/iofq_test.py` uses `with_banks`.
2. σ_λ from the lookup table (essreduce).
   - `ess/reduce/unwrap/lut.py` stores `variances` on the lookup table (`make_wavelength_lut_from_simulation`, and the analytical path around `_polygon_intersections`).
   - `WavelengthInterpolator` in `to_wavelength.py` only interpolates values. We need σ_λ as a function of (Ltotal, λ) for the dense denominator grid, not per event. One option: interpolate stddev on the (distance, event_time_offset) table, then map to wavelength via the mean wavelength of the same table.
   - Note `mask_large_uncertainty_in_lut` replaces uncertain entries with NaN; decide how the resolution handles NaN regions (they are masked in the data anyway).
3. Resolution sums in esssans.
   - Compute σ_i² on `QDetector[RunType, Denominator]` (pixel × λ midpoints, has `Q` and can give `two_theta`, `L2`, `Ltotal` via the graph).
   - Histogram `N·Q` and `N·(σ² + Q²)` over Q exactly like the denominator (`_bin_in_q`). A `moment` dimension of size 2 may let this reuse `mask_and_scale_wavelength_q`, `_reduce` and the bank/run merge unchanged; check whether a new `IofQPart`-like key or separate domain types is cleaner.
   - Final provider: σ_bin² from S0, S1, S2, then attach as variances of the `Q` point coordinate on I(Q). `save_background_subtracted_iofq` in `io.py` already writes Q variances as `resolutions`. Background-subtracted I(Q) uses the sample run's resolution.
   - Only QBins (1D) initially; Qx/Qy resolution is out of scope.
4. Optional: mixture components (classes by fixed σ/Q log bins × Q sub-bins, ~10–20 per bin).

## Parameters still open

R1, R2, L1, ΔR for LoKI (NeXus or user input), bin center vs `Q_mean` as the reported Q, ΔR for tilted straws (radial pixel extent from the pixel shape), gravity. See the open questions in the issue; get answers from the instrument scientists before hard-coding anything.

## Toy model

A JavaScript toy model (two flat banks, sphere form factor, long pulse) produced the numbers in the issue. It is not in the repository. It lives in a private explainer page that Simon can open: https://claude.ai/artifact/JhbFrDS2YKZMU9ZMc7i3FP (the page loads `sim.js`). Use it only for intuition; tests should use the real workflow with LoKI test data. A good test: for a single bank, merged vs. unmerged S0/S1/S2 must agree, and σ_bin² must equal a brute-force N-weighted second moment computed from the dense (pixel, λ) arrays.

## Environment

Worktree `.worktrees/360-q-resolution` (named before the issue existed), branch `763-q-resolution`. Use `pixi run -e esssans ...`; `uv` is not installed in the devcontainer.
