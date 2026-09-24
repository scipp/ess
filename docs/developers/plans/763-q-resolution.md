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

Bank merging (1) and the resolution work (2, 3) are independent and can be done in either order; see "Independence from bank merging" below.

1. Bank merging (separate PR).
   - `ess/sans/workflow.py`: `with_banks` currently maps over `NeXusDetectorName` without reducing, and the `parameter_mappers[NeXusDetectorName]` line is commented out with a TODO.
   - Reduce at `NormalizedQ[RunType, IofQPart]` for all parts, using `merge_contributions`, the same pattern `_set_runs` uses for multiple runs. Write it as a loop over the parts so a `ResolutionMoments` part (step 3) is included automatically.
   - Check that per-bank `DetectorBankSizes` and `DimsToKeep` still produce identical output dims across banks.
   - Update the `with_banks` docstring (the "different Q-resolution" argument).
   - Existing tests: `tests/loki/iofq_test.py` uses `with_banks`.
2. σ_λ from the lookup table (essreduce).
   - `ess/reduce/unwrap/lut.py` stores `variances` on the lookup table (`make_wavelength_lut_from_simulation`, and the analytical path around `_polygon_intersections`).
   - `WavelengthInterpolator` in `to_wavelength.py` only interpolates values. We need σ_λ as a function of (Ltotal, λ) for the dense denominator grid, not per event. One option: interpolate stddev on the (distance, event_time_offset) table, then map to wavelength via the mean wavelength of the same table.
   - Note `mask_large_uncertainty_in_lut` replaces uncertain entries with NaN; decide how the resolution handles NaN regions (they are masked in the data anyway).
3. Resolution sums in esssans.
   - Compute σ_i² on `QDetector[RunType, Denominator]` (pixel × λ midpoints, has `Q` and can give `two_theta`, `L2`, `Ltotal` via the graph).
   - Make the sums a third member of `IofQPart`: `IofQPart = TypeVar('IofQPart', Numerator, Denominator, ResolutionMoments)`, carrying `N·Q` and `N·(σ² + Q²)` along a size-2 `moment` dimension. Generic providers (`compute_Q`, `bin_in_q`, `reduce_q`) then apply unchanged.
   - New code: the provider creating the sums from the denominator grid, and a variant of `mask_and_scale_wavelength_q` that multiplies by the monitor term.
   - Add `ResolutionMoments` to the `for part in (Numerator, Denominator)` loop in `_set_runs`.
   - Final provider: σ_bin² from S0, S1, S2, then attach as variances of the `Q` point coordinate on I(Q). `save_background_subtracted_iofq` in `io.py` already writes Q variances as `resolutions`. Background-subtracted I(Q) uses the sample run's resolution.
   - Only QBins (1D) initially; Qx/Qy resolution is out of scope.
4. Optional: mixture components, ~10–20 per bin. Class = (Q sub-bin, σ/Q class); σ/Q class k holds elements with `f^k ≤ σ_i/Q_i < f^(k+1)` (f = 1.5 worked in the toy model); Q sub-bins split each Q bin into n equal parts (n = 4). Edges must be fixed in advance so sums merge.

## Independence from bank merging

- Compute `σ² = S2/S0 − (S1/S0)²` only once, after all merging. Never compute σ per bank or run and average.
- Every merge point must merge the resolution sums too. Today that is only `_set_runs`. The instrument scientists will probably process runs separately and merge the files instead, but the multi-run workflow must stay correct.
- Test for both: σ from two runs merged by the workflow equals σ computed from the summed S0, S1, S2 of the individual runs.

## Parameters still open

R1, R2, L1, ΔR for LoKI (NeXus or user input), bin center vs `Q_mean` as the reported Q, ΔR for tilted straws (radial pixel extent from the pixel shape), gravity. See the open questions in the issue; get answers from the instrument scientists before hard-coding anything.

## Toy model

A JavaScript toy model (two flat banks, sphere form factor, long pulse) produced the numbers in the issue. It is not in the repository. It lives in a private explainer page that Simon can open: https://claude.ai/artifact/JhbFrDS2YKZMU9ZMc7i3FP (the page loads `sim.js`). Use it only for intuition; tests should use the real workflow with LoKI test data. A good test: for a single bank, merged vs. unmerged S0/S1/S2 must agree, and σ_bin² must equal a brute-force N-weighted second moment computed from the dense (pixel, λ) arrays.

## Environment

Worktree `.worktrees/360-q-resolution` (named before the issue existed), branch `763-q-resolution`. Use `pixi run -e esssans ...`; `uv` is not installed in the devcontainer.
