"""Builds amor-direct-beam-reduction.ipynb (no outputs).  Edit the cells here, then run
    python build_notebook.py
and execute the notebook where the raw files are."""
import json

cells = []
def md(text): cells.append(dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(keepends=True)))
def code(text): cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.strip("\n").splitlines(keepends=True)))

md("""
# Direct-beam reduction for Amor

This notebook reduces reflectometry data from the PSI instrument [Amor](https://www.psi.ch/en/sinq/amor), measured in
divergent-beam mode and **normalised with a direct beam**, using [Scipp](https://scipp.github.io).

It goes step by step from the event files to $R(Q)$:

1. load the events of a run
2. look at the detector images
3. turn events into wavelength and histogram them on (wavelength, detector row), for the direct beam and for the sample
4. assign an angle and a $Q$ to every (wavelength, detector row) cell
5. divide and project onto $Q$
6. reduce several angles and save the result

The algorithm follows the direct-beam mode (`-nm d`) of the PSI program *eos* 3.3.4, and the last section compares the two.
The functions live in `amor_scipp.py` next to this notebook; they use `scipp`, `scippnexus` and the detector geometry of
`ess.amor`.

It differs from the [divergent data reduction for Amor](amor-reduction.ipynb) notebook in three ways: the reference is a direct beam
instead of a supermirror (no supermirror reflectivity, no footprint correction, and the reference map is mirrored on the detector),
each pulse is weighted by the proton current, and the file layout is the one written by Amor since 2026.
""")

md("## Setup")
code("""
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipp as sc

import amor_scipp as amor
""")

md("""
### What to reduce

The example is a silicon block in D₂O measured at three angles with a 0.12° divergent beam.
Each angle has its own direct-beam run. Runs are given as in eos: `"6310-6312"` means the three runs are added.

The raw files of this example are not distributed with the notebook. Point `raw_path` (or the environment variable
`AMOR_RAW_PATH`) at a folder with your own Amor files and change the run numbers and settings below.
""")
code("""
raw_path = os.environ.get("AMOR_RAW_PATH", "raw")           # folder with amor2026n00NNNN.hdf

measurements = {                       # name: (sample runs, direct-beam run)
    "first angle":  ("6309",      "6327"),
    "second angle": ("6310-6312", "6328"),
    "third angle":  ("6313-6317", "6329"),
}

config = amor.make_config(
    q_resolution=0.05,                 # relative width of the Q bins
    q_range=(0.005, 0.27),             # 1/angstrom; events outside are dropped
    lambda_range=(3.0, 12.0),          # angstrom
    y_range=(18, 48),                  # detector strips kept (across the beam)
    mu_offset=-0.055,                  # deg, correction to the sample angle
    theta_range_rel=(-0.055, 0.06),    # deg, accepted angles relative to the beam centre
)
""")

md("""
## 1. Load a run

`load_run` reads the event list with `scippnexus` and returns it as a flat table: one row per neutron, with the time since the
pulse (`event_time_offset`), the detector pixel (`event_id`) and the time stamp of the pulse it belongs to.
It also returns the proton-current log and the pulse times. These files contain no chopper trigger signal, so the pulse
times are taken from the beam-monitor group.
""")
code("""
sample_files = amor.run_files(measurements["first angle"][0], raw_path)
norm_files = amor.run_files(measurements["first angle"][1], raw_path)

run = amor.load_run(sample_files[0])
run["events"]
""")
code("""
p = run["par"]
print(f"sample angle mu = {p['mu']:.3f} deg, detector angle nu = {p['nu']:.3f} deg, divergence = {p['div']:.2f} deg")
print(f"chopper {p['chopper_speed']:.0f} rpm -> pulse length tau = {p['tau']*1e3:.2f} ms; chopper to detector {p['detector_distance']-p['chopper_distance']:.0f} mm")
print(f"{run['pulses'].sizes['pulse']} pulses, {run['events'].sizes['event']} events")
""")
code("""
pc = run["proton_current"]
t = (pc.coords["time"].values - pc.coords["time"].values[0]) / 1e9
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.plot(t, pc.values, lw=0.8)
ax.axhline(config["low_current"], color="C3", ls="--", label="pulses below this are dropped")
ax.set_xlabel("time (s)"); ax.set_ylabel("proton current (µA)"); ax.legend();
""")

md("""
## 2. Detector images

The detector has 14 blades of 32 wires, read out in 64 strips. `pixel_table` gives, for every pixel, its strip (across the beam),
its row (wire index, along the scattering angle), its angle relative to the detector centre and the extra flight path inside
the detector. The angles come from `ess.amor.geometry`.

Each row corresponds to one scattering angle, so the image of the reflected beam is a map of intensity against angle.
The direct beam hits the same rows, but upside down.
""")
code("""
pixels = amor.pixel_table("eos")
image_norm = amor.detector_image(norm_files[0], pixels)
image_sample = amor.detector_image(sample_files[0], pixels)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, image, title in zip(axes, (image_norm, image_sample), ("direct beam", "sample")):
    im = ax.imshow(image.values, origin="lower", aspect="auto", norm=LogNorm(vmin=1, vmax=image.values.max()),
                   extent=(-0.5, 63.5, -0.5, 447.5))
    ax.axvline(config["y_range"][0] - 0.5, color="w", ls="--"); ax.axvline(config["y_range"][1] + 0.5, color="w", ls="--")
    ax.set_ylim(190, 258); ax.set_xlabel("strip (across the beam)"); ax.set_ylabel("row (scattering angle)")
    ax.set_title(f"{title}: {int(image.values.sum()):,} events"); fig.colorbar(im, ax=ax, label="counts")
""")

md("""
## 3. From events to wavelength maps

For every event:

* the time of flight is shifted by the chopper phase and folded into one frame;
* it is corrected for the angle at which the neutron passed the chopper opening;
* the wavelength follows from the flight path between chopper and pixel.

Events are then dropped if their pulse had too little proton current, if their strip or wavelength is outside the chosen range,
or (sample only) if their $Q$ is outside `q_range`. What remains is histogrammed on wavelength and detector row.

The **monitor** of a run is the proton current at each accepted pulse times the pulse length, summed.

### Direct beam

The direct-beam map is mirrored along the detector rows, so that each row of the sample map is divided by the row that saw the
same incoming ray.
""")
code("""
norm_map, norm_monitor = amor.direct_beam_map(norm_files, config, pixels)
print(f"direct beam: {norm_map.values.sum():,.0f} events kept, monitor {norm_monitor:.1f}")
norm_map
""")
md("### Sample")
code("""
sample_map, monitor, mu, nu, n_kept = amor.sample_map(sample_files, config, pixels)
print(f"sample: {n_kept:,} events kept, monitor {monitor:.1f}; mu = {mu:.3f} deg (offset applied), nu = {nu:.3f} deg")
""")
code("""
lam_edges = norm_map.coords["wavelength"].values
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, m, title in zip(axes, (norm_map, sample_map), ("direct beam N(λ, row), mirrored", "sample I(λ, row)")):
    v = m.data.transpose(["wavelength", "z_index"]).values.T
    im = ax.pcolormesh(lam_edges, np.arange(449) - 0.5, np.where(v > 0, v, np.nan), norm=LogNorm(vmin=1, vmax=v.max()))
    ax.set_ylim(190, 258); ax.set_xlim(2.5, 13); ax.set_xlabel("wavelength (Å)"); ax.set_ylabel("row"); ax.set_title(title)
    fig.colorbar(im, ax=ax, label="counts per cell")
""")
md("""
In the sample map the bright band that runs diagonally is the edge of total reflection: it moves to shorter wavelength at lower angle.
""")

md("""
## 4. Angle and Q of every cell

Each detector row has a fixed angle relative to the detector centre. The reflection angle of a cell is

$$\\theta = \\nu - \\mu + \\delta_\\mathrm{row} + \\text{gravity term}(\\lambda)$$

and $Q = 4\\pi \\sin\\theta / \\lambda$. Slow neutrons fall measurably on the way to the detector, which is what the gravity term corrects.

A cell is **accepted** if its angle is inside `theta_range_rel` around $\\nu - \\mu$ and its wavelength is inside `lambda_range`.
""")
code("""
grid = amor.lz_grid(mu, nu, amor.load_par_detector_distance(sample_files[0]), config, pixels)
accepted = grid["accepted"].values
rows = np.where(accepted.any(axis=0))[0]
print(f"beam centre at theta = {grid['theta0']:.3f} deg; accepted window {grid['theta_window'][0]:.3f} to {grid['theta_window'][1]:.3f} deg")
print(f"{accepted.sum()} cells accepted, detector rows {rows.min()} to {rows.max()}")
""")
md("""
Dividing the two maps cell by cell, and scaling by the ratio of the monitors, gives the reflectivity of each cell.
Lines of constant $Q$ are drawn on top: the reflectivity should only change across them, not along them.
""")
code("""
I = sample_map.data.transpose(["wavelength", "z_index"]).values
N = norm_map.data.transpose(["wavelength", "z_index"]).values
with np.errstate(divide="ignore", invalid="ignore"):
    R_cell = np.where(accepted & (N > 0) & (I > 0), I / N * norm_monitor / monitor, np.nan)

from matplotlib.collections import PolyCollection
theta = grid["theta"].values
pitch = float(np.median(np.abs(np.diff(pixels["delta_z"].values))))
li, zi = np.nonzero(np.isfinite(R_cell))
quads = np.stack([np.c_[lam_edges[li], theta[li, zi] - pitch / 2], np.c_[lam_edges[li + 1], theta[li, zi] - pitch / 2],
                  np.c_[lam_edges[li + 1], theta[li, zi] + pitch / 2], np.c_[lam_edges[li], theta[li, zi] + pitch / 2]], axis=1)
fig, ax = plt.subplots(figsize=(8, 4.5))
coll = PolyCollection(quads, array=R_cell[li, zi], cmap="magma", norm=LogNorm(vmin=5e-3, vmax=2), edgecolors="face", linewidths=0.2)
ax.add_collection(coll); ax.set_facecolor("0.15")
lam_line = np.linspace(*config["lambda_range"], 100)
for q in config["q_edges"].values[::2]:
    ax.plot(lam_line, np.rad2deg(np.arcsin(np.clip(q * lam_line / (4 * np.pi), -1, 1))), color="w", lw=0.5)
ax.set_xlim(*config["lambda_range"]); ax.set_ylim(theta[accepted].min() - 0.005, theta[accepted].max() + 0.005)
ax.set_xlabel("wavelength (Å)"); ax.set_ylabel("reflection angle θ (deg)")
fig.colorbar(coll, ax=ax, label="R per cell");
""")

md("""
## 5. Project onto Q

The accepted cells are sorted into $Q$ bins. In each bin

$$R(Q) = \\frac{\\sum I}{\\sum N} \\cdot \\frac{\\text{monitor}_\\text{direct beam}}{\\text{monitor}_\\text{sample}}$$

The $Q$ grid is linear below 0.01 Å⁻¹ and geometric above, with relative step `q_resolution`.
The resolution $\\sigma_Q$ of a bin is the average of the cell resolutions, weighted with the direct-beam counts.
""")
code("""
result, I_q, W_q, N_q = amor.project_to_q(sample_map, norm_map, grid, monitor, norm_monitor, config)
r_first = amor.reflectivity_data_array(result)
r_first
""")
code("""
def plot_r(curves, ax=None, rq4=False):
    ax = ax or plt.subplots(figsize=(7, 4.2))[1]
    for label, da in curves.items():
        q, r, e = da.coords["Q"].values, da.values, np.sqrt(da.variances)
        f = q**4 if rq4 else 1.0
        ax.errorbar(q, r * f, e * f, xerr=da.coords["Q_resolution"].values, marker="o", ms=3, lw=0.8, elinewidth=0.6, label=label)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Q (Å⁻¹)"); ax.set_ylabel("R·Q⁴ (Å⁻⁴)" if rq4 else "R"); ax.legend()
    return ax

plot_r({"first angle": r_first});
""")

md("""
## 6. All angles in one go

`reduce` runs steps 3 to 5 for one curve. Each angle is reduced against its own direct beam.
The curves are **not** scaled or stitched: where they overlap, any offset between them is a property of the normalisation
and is left visible.
""")
code("""
curves, info = {}, {}
for name, (sample_runs, norm_run) in measurements.items():
    data, info[name] = amor.reduce(amor.run_files(sample_runs, raw_path), amor.run_files(norm_run, raw_path), config)
    curves[name] = amor.reflectivity_data_array(data)
    print(f"{name}: {info[name]['events_kept']:,} events, monitor {info[name]['monitor_sample']:.0f}, theta = {info[name]['nu'] - info[name]['mu']:.3f} deg")
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
plot_r(curves, ax=axes[0]); plot_r(curves, ax=axes[1], rq4=True);
""")

md("""
### Save

One text file per angle with the columns $Q$, $R$, $\\sigma_R$, $\\sigma_Q$.
$\\sigma_R$ uses the eos formula $R\\sqrt{1/(I+0.1) + 1/(N+0.1)}$; the array returned by `reduce` also holds the plain Poisson
estimate in a fifth column.
""")
code("""
os.makedirs("reduced", exist_ok=True)
for name, da in curves.items():
    fname = os.path.join("reduced", name.replace(" ", "_") + ".Rqz.dat")
    np.savetxt(fname, np.c_[da.coords["Q"].values, da.values, np.sqrt(da.variances), da.coords["Q_resolution"].values],
               header=f"Amor direct-beam reduction with scipp {sc.__version__}; {measurements[name]}\\nQz (1/angstrom)  R  sR  sQz")
    print("wrote", fname)
""")

md("""
## Comparison with eos

If the eos results for the same runs are available, the two can be compared bin by bin. With the default conventions the
agreement is at the level of 10⁻⁷; the remainder is a constant factor per curve that comes from eos storing the per-pulse monitor
in single precision.
""")
code("""
eos_path = os.environ.get("AMOR_EOS_PATH", "eos")           # folder with the eos .Rqz.ort files
eos_files = {"first angle": "Si-D2O-n_q008.Rqz.ort", "second angle": "Si-D2O-n_q024.Rqz.ort", "third angle": "Si-D2O-n_q066.Rqz.ort"}

if os.path.isdir(eos_path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for name, f in eos_files.items():
        e = np.loadtxt(os.path.join(eos_path, f)); e = e[np.isfinite(e[:, 1])]
        da = curves[name]
        assert np.allclose(e[:, 0], da.coords["Q"].values), "different Q grids"
        ok = e[:, 1] > 0
        ax.plot(e[ok, 0], (da.values[ok] / e[ok, 1] - 1) * 1e6, marker="o", ms=3, label=name)
        print(f"{name}: largest relative difference in R {np.abs(da.values[ok] / e[ok, 1] - 1).max():.1e}")
    ax.set_xscale("log"); ax.set_xlabel("Q (Å⁻¹)"); ax.set_ylabel("(scipp / eos − 1) × 10⁶"); ax.legend()
else:
    print("no eos results found at", eos_path)
""")

md("""
## Conventions that differ from `ess.amor`

`make_config(convention="ess")` switches three choices to the ones made in the `ess.amor` workflow:

| choice | eos | `ess.amor` |
|---|---|---|
| flight path inside the detector | `wire · dX` | `(nWires − 1 − wire) · dX` |
| $h/m_n$ | constant hard-coded in eos | `scipp.constants` |
| gravity | $\\theta + \\arctan(3.07\\times10^{-10}\\,L\\,\\lambda^2)$ | $\\arcsin(\\sin\\theta + g L \\lambda^2 m_n^2 / 2h^2)$ |

The first one matters most: it changes each wavelength by up to 0.65 %, more than the width of a wavelength bin.
The cell below shows the effect on the second angle.
""")
code("""
name = "second angle"
files = amor.run_files(measurements[name][0], raw_path), amor.run_files(measurements[name][1], raw_path)
kw = dict(q_resolution=0.05, q_range=(0.005, 0.27), lambda_range=(3.0, 12.0), mu_offset=-0.055, theta_range_rel=(-0.055, 0.06))
fig, ax = plt.subplots(figsize=(7, 3.2))
for part in ("detx", "hdm", "gravity"):
    data, _ = amor.reduce(*files, amor.make_config(convention="ess", ess_parts=(part,), **kw))
    alt = amor.reflectivity_data_array(data)
    ax.plot(alt.coords["Q"].values, (alt.values / curves[name].values - 1) * 100, marker="o", ms=3, label=part)
ax.set_xscale("log"); ax.set_xlabel("Q (Å⁻¹)"); ax.set_ylabel("change of R (%)"); ax.legend(title="convention switched");
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3 (ipykernel)", language="python", name="python3"),
                                      language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open("amor-direct-beam-reduction.ipynb", "w"), indent=1, ensure_ascii=False)
print("wrote amor-direct-beam-reduction.ipynb with", len(cells), "cells")
