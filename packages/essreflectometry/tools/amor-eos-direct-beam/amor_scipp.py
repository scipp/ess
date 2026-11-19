# SPDX-License-Identifier: BSD-3-Clause
"""Amor reduction in scipp, following the algorithm of eos 3.3.4 (direct-beam normalisation).

Uses
  * scippnexus to load the NXevent_data and NXlog groups of the October-2026 file layout,
  * ess.amor.geometry (essreflectometry) for the multiblade detector pixel table,
  * scipp variables with units for every per-event quantity, and scipp histogramming for
    the (wavelength, detector-z) map and for the projection onto Q.

The steps and constants mirror eos so that the result can be compared number by number:

  events:   tof = event_time_offset, shifted by the chopper trigger phase, folded into one
            frame, corrected for the chopper opening seen by each detector angle
            -> wavelength from the flight path chopper->pixel
  filters:  pulses with proton current below threshold, strips (y) outside range,
            wavelength outside range, event Q outside the requested Q range
  monitor:  proton current at each pulse x pulse length, summed over accepted pulses
  maps:     I(lambda, z) for the sample, N(lambda, z) for the direct beam (flipped in z,
            because the direct beam hits the detector mirrored)
  grid:     theta(lambda, z) = nu - mu + delta_z + gravity term;  Q = 4 pi sin(theta)/lambda
  R(Q):     sum I / sum N over the unmasked (lambda, z) cells falling in each Q bin,
            times monitor_norm / monitor_sample

CONVENTION switches the few places where essreflectometry's own Amor code differs from
eos ("eos" reproduces eos; "ess" uses the ess.amor choices):
  * path length inside the detector: eos wire*dX, ess (nWires-1-wire)*dX
  * h/m_n: eos hard-coded 1980s CODATA value, ess scipp.constants
  * gravity: eos atan(3.07e-10 * L * lambda^2) added to the angle, ess asin(sin + g term)
"""
import argparse
import os

import h5py
import numpy as np
import scipp as sc
import scippnexus as snx
from ess.amor.geometry import Detector, pixel_coordinates_in_detector_system

HDM_EOS = 6.626176e-34 / 1.674928e-27            # h/m_n as hard-coded in eos, m^2/s
LAMBDA_CUT, LAMBDA_MAX, DLDL = 2.5, 15.0, 0.005   # eos const / LZGrid


# ------------------------------------------------------------------------------- loading
def load_run(path):
    """One raw file -> dict(events, pulses, proton_current, par).  events is a flat scipp
    table (dim 'event') with coords event_time_offset [s], event_id, pulse_time [ns]."""
    with snx.File(path) as f:
        binned = f["entry1/detector/data"][()]                    # binned by pulse
    binned.bins.coords["pulse_time"] = sc.bins_like(binned, binned.coords["event_time_zero"])
    ev = binned.bins.constituents["data"].copy()
    ev = sc.DataArray(
        sc.ones(dims=["event"], shape=[ev.sizes[ev.dim]], unit="counts"),
        coords={
            "event_time_offset": ev.coords["event_time_offset"].rename_dims({ev.dim: "event"}).to(unit="s", dtype="float64"),
            "event_id": ev.coords["event_id"].rename_dims({ev.dim: "event"}),
            # scippnexus gives datetime64[ns]; keep the epoch integer, as the monitor log uses
            "pulse_time": sc.array(dims=["event"], values=ev.coords["pulse_time"].values.astype("int64"), unit="ns"),
        })
    with h5py.File(path, "r") as h:
        g = lambda k: float(np.ravel(h[k][()])[0])
        par = dict(mu=g("entry1/measurement_configuration/mu"), nu=g("entry1/measurement_configuration/nu"),
                   div=g("entry1/measurement_configuration/divergence"),
                   chopper_distance=g("entry1/Amor/chopper/distance"),
                   detector_distance=g("entry1/Amor/detector_stage/transformation/distance"),
                   chopper_speed=g("entry1/Amor/chopper/rotation_speed"), chopper_phase=g("entry1/Amor/chopper/phase"))
        n_trig = h["entry1/detector/trigger/event_time_zero"].shape[0]
        bm_time = h["entry1/beam_monitor/time"][:].astype("int64")
        pc_val = h["entry1/detector/proton_current/value"][:].astype("float64").reshape(len(h["entry1/detector/proton_current/time"]), -1)[:, 0]
        pc_time = h["entry1/detector/proton_current/time"][:].astype("int64")
    if n_trig > 2:
        raise NotImplementedError("file has chopper triggers; eos then takes pulse times and tau from them")
    par["tau"] = 30.0 / par["chopper_speed"]                       # s, half a chopper turn
    pulses = sc.array(dims=["pulse"], values=bm_time, unit="ns")   # eos: beam-monitor times
    proton_current = sc.DataArray(sc.array(dims=["time"], values=pc_val, unit="uA"),
                                  coords={"time": sc.array(dims=["time"], values=pc_time, unit="ns")})
    return dict(events=ev, pulses=pulses, proton_current=proton_current, par=par, file=os.path.basename(path))


def pixel_table(convention):
    """Flat per-pixel table indexed by event_id-1: strip, z_index, delta [deg], detX [mm]."""
    px = pixel_coordinates_in_detector_system()
    flat = px.flatten(to="pixel")
    wire = sc.broadcast(px.coords["wire"], sizes=px.sizes).flatten(to="pixel")
    strip = sc.broadcast(px.coords["strip"], sizes=px.sizes).flatten(to="pixel")
    z = sc.broadcast(px.coords["z_index"], sizes=px.sizes).flatten(to="pixel")
    delta = sc.broadcast(px.coords["pixel_divergence_angle"], sizes=px.sizes).flatten(to="pixel").to(unit="deg")
    if convention == "eos":
        detx = (wire * Detector.dX).to(unit="mm", dtype="float64")
    else:
        detx = sc.broadcast(px.coords["distance_in_detector"], sizes=px.sizes).flatten(to="pixel").to(unit="mm")
    assert np.array_equal(flat.values, np.arange(1, flat.sizes["pixel"] + 1))
    delta_z = px.coords["pixel_divergence_angle"].to(unit="deg")["strip", 0] if "strip" in px.coords["pixel_divergence_angle"].dims \
        else px.coords["pixel_divergence_angle"].to(unit="deg")
    delta_z = sc.broadcast(delta_z, sizes={"blade": px.sizes["blade"], "wire": px.sizes["wire"]}).flatten(to="z_index")
    return dict(strip=strip, z=z, delta=delta, detx=detx, delta_z=delta_z)


# ----------------------------------------------------------------------- event processing
def process_events(run, cfg, px, mu_offset=0.0, q_range=None):
    """Applies the eos event chain; returns (I(lambda, z) histogram, monitor, mu, nu, n kept)."""
    ev, par = run["events"], run["par"]
    hdm = (sc.constants.h / sc.constants.m_n).to(unit="m^2/s").value if "hdm" in cfg["parts"] else HDM_EOS
    tau = sc.scalar(par["tau"], unit="s")
    ch1 = cfg["chopper_phase_offset"]                             # eos replaces ch1TriggerPhase by this
    cdd = sc.scalar(par["detector_distance"] - par["chopper_distance"], unit="mm")
    mu = par["mu"] + mu_offset
    nu = par["nu"]

    # monitor: proton current valid at each pulse (previous log entry), low-current pulses rejected
    pc = run["proton_current"]
    lookup = sc.lookup(pc, "time", mode="previous", fill_value=sc.scalar(0.0, unit="uA"))
    cur = lookup[run["pulses"]]
    good = cur > sc.scalar(cfg["low_current"], unit="uA")
    monitor = (sc.where(good, cur, sc.scalar(0.0, unit="uA")) * (2 * tau)).sum().to(unit="uA*s").value * 1e-3
    good_times = run["pulses"][good].values
    keep = sc.array(dims=["event"], values=np.isin(ev.coords["pulse_time"].values, good_times))

    # time of flight -> one frame
    tof = ev.coords["event_time_offset"] + tau * ((ch1 - par["chopper_phase"] / 2) / 180)
    keep &= tof <= 2 * tau
    tof_cut = (sc.scalar(LAMBDA_CUT * 1e-13 / hdm, unit="s/mm") * cdd).to(unit="s")
    total_offset = tof_cut + tau * ((ch1 + par["chopper_phase"] / 2) / 180)
    tof = ((tof - (tof_cut - tau)) % tau) + total_offset

    # pixel geometry
    idx = ev.coords["event_id"].values.astype(np.int64) - 1
    pick = lambda v: sc.array(dims=["event"], values=v.values[idx], unit=v.unit)
    strip, z, delta, detx = pick(px["strip"]), pick(px["z"]), pick(px["delta"]), pick(px["detx"])
    keep &= (strip >= sc.scalar(cfg["y_range"][0], unit=strip.unit)) & (strip <= sc.scalar(cfg["y_range"][1], unit=strip.unit))

    # chopper-opening correction and wavelength
    tof = tof - (delta / sc.scalar(180.0, unit="deg")) * tau
    wavelength = (sc.scalar(1e13 * hdm, unit="angstrom*mm/s") * tof / (cdd + detx)).to(unit="angstrom")
    lo, hi = (sc.scalar(v, unit="angstrom") for v in cfg["lambda_range"])
    keep &= (wavelength >= lo) & (wavelength <= hi)

    # event Q (no gravity, as in eos) only used for the optional Q-range filter
    if q_range is not None:
        theta_e = (sc.scalar(nu - mu, unit="deg") + delta).to(unit="rad")
        q_e = 4 * np.pi * sc.sin(theta_e) / wavelength
        keep &= (q_e >= sc.scalar(q_range[0], unit="1/angstrom")) & (q_e <= sc.scalar(q_range[1], unit="1/angstrom"))

    table = sc.DataArray(sc.where(keep, ev.data, sc.scalar(0.0, unit="counts")),
                         coords={"wavelength": wavelength, "z_index": z.to(dtype="float64")})
    hist = table.hist(wavelength=cfg["lambda_edges"], z_index=cfg["z_edges"])
    return sc.values(hist), monitor, mu, nu, int(keep.values.sum())


def lambda_edges():
    n = int(np.log(LAMBDA_MAX / LAMBDA_CUT) / np.log(1 + DLDL) + 1)
    return sc.array(dims=["wavelength"], values=LAMBDA_CUT * (1 + DLDL) ** np.arange(n), unit="angstrom")


def q_edges(resolution, q_range):
    """The eos Q grid: linear below 0.01, geometric above."""
    qq = 0.01
    grid = np.arange(0, qq, qq * resolution)
    grid = np.append(grid, qq * (1.0 + resolution) ** np.arange(int(np.log(q_range[1] / qq) / np.log(1 + resolution))))
    return sc.array(dims=["Q"], values=grid[grid >= q_range[0]], unit="1/angstrom")


# ------------------------------------------------------------------------------ reduction
def reduce(sample_files, norm_files, cfg):
    px = pixel_table("ess" if "detx" in cfg["parts"] else "eos")
    cfg = dict(cfg, lambda_edges=lambda_edges(),
               z_edges=sc.arange("z_index", 0.0, Detector.nBlades.value * Detector.nWires.value + 1.0, unit=None))
    cfg["z_edges"].unit = px["z"].unit

    norm, norm_mon = None, 0.0
    for f in norm_files:
        h, m, *_ = process_events(load_run(f), cfg, px)
        norm = h if norm is None else norm + h
        norm_mon += m
    # the direct beam lands mirrored on the detector with respect to the reflected beam
    norm = sc.DataArray(sc.array(dims=norm.dims, values=norm.values[:, ::-1], unit=norm.unit), coords=dict(norm.coords))

    inten, mon, nkept, mu, nu = None, 0.0, 0, None, None
    for f in sample_files:
        h, m, mu_f, nu_f, n = process_events(load_run(f), cfg, px, mu_offset=cfg["mu_offset"], q_range=cfg["q_filter"])
        inten = h if inten is None else inten + h
        mon += m; nkept += n
        if mu is None:
            # as in eos: each run's own angles are used for its event filter, but the
            # (lambda, z) -> Q grid of the summed map uses the FIRST run's angles
            mu, nu = mu_f, nu_f

    # (lambda, z) grid: angle with gravity term, Q, resolution
    lam = sc.midpoints(cfg["lambda_edges"])
    t0 = nu - mu
    alpha = sc.scalar(t0, unit="deg") + px["delta_z"]                          # dim z_index
    L = sc.scalar(load_par_detector_distance(sample_files[0]), unit="mm")
    if "gravity" not in cfg["parts"]:
        grav = sc.atan(sc.scalar(3.07e-10, unit="1/(mm*angstrom^2)") * L * lam**2).to(unit="deg")
        alpha_lz = alpha + grav
    else:
        c = (sc.constants.g * sc.constants.m_n**2 / sc.constants.h**2 / 2)
        alpha_lz = sc.asin((c * L * lam**2).to(unit="dimensionless") + sc.sin(alpha.to(unit="rad"))).to(unit="deg")
    alpha_lz = alpha_lz.transpose(["wavelength", "z_index"])
    q_lz = 4 * np.pi * sc.sin(alpha_lz.to(unit="rad")) / lam
    if q_lz["wavelength", 0]["z_index", q_lz.sizes["z_index"] // 2].value < 0:
        q_lz = -q_lz
    res_lz = q_lz * sc.sqrt(sc.scalar(0.022**2) + (sc.scalar(0.008, unit="deg") / alpha_lz) ** 2)

    T = cfg["theta_range_rel"]
    mask = (alpha_lz >= sc.scalar(t0 + T[0], unit="deg")) & (alpha_lz <= sc.scalar(t0 + T[1], unit="deg"))
    mask = mask & (lam >= sc.scalar(cfg["lambda_range"][0], unit="angstrom")) & (lam <= sc.scalar(cfg["lambda_range"][1], unit="angstrom"))
    w = sc.where(mask, sc.scalar(1.0), sc.scalar(0.0))

    def to_q(values):
        da = sc.DataArray((values * w).copy(), coords={"Q": q_lz.to(unit="1/angstrom")})
        return da.flatten(to="cell").hist(Q=cfg["q_edges"]).data
    I_q = to_q(inten.data.transpose(["wavelength", "z_index"]))
    W_q = to_q(norm.data.transpose(["wavelength", "z_index"]))
    N_q = to_q(sc.ones(sizes=q_lz.sizes))
    WR_q = to_q(norm.data.transpose(["wavelength", "z_index"]) * res_lz.to(unit="1/angstrom"))

    I, W, N, WR = I_q.values, np.maximum(W_q.values, 1e-10), N_q.values, WR_q.values
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(N > 0, I * norm_mon / mon / W, np.nan)
        dR_eos = np.where(N > 0, R * np.sqrt(1.0 / (I + 0.1) + 1.0 / (W + 0.1)), np.nan)        # eos formula
        dR_poisson = np.where((N > 0) & (I > 0), R * np.sqrt(1.0 / np.maximum(I, 1e-300) + 1.0 / W), np.nan)
        dQ = np.where(N > 0, WR / W, np.nan)
    Q = sc.midpoints(cfg["q_edges"]).values
    info = dict(mu=mu, nu=nu, monitor_sample=mon, monitor_norm=norm_mon, events_kept=nkept,
                counts_in_map=float(inten.values.sum()), norm_counts=float(norm.values.sum()))
    return np.c_[Q, R, dR_eos, dQ, dR_poisson], info


def load_par_detector_distance(path):
    with h5py.File(path, "r") as h:
        return float(np.ravel(h["entry1/Amor/detector_stage/transformation/distance"][()])[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-f", required=True, help="sample runs, e.g. 6310-6312")
    ap.add_argument("-n", required=True, help="direct-beam runs")
    ap.add_argument("-o", required=True, help="output name (without .Rqz.ort)")
    ap.add_argument("-rp", default="raw"); ap.add_argument("-Y", default="2026")
    ap.add_argument("-r", type=float, default=0.01); ap.add_argument("-q", type=float, nargs=2, default=None)
    ap.add_argument("-l", type=float, nargs=2, default=[3, 12.5]); ap.add_argument("-y", type=int, nargs=2, default=[18, 48])
    ap.add_argument("-m", type=float, default=0.0); ap.add_argument("-T", type=float, nargs=2, default=[-0.75, 0.75])
    ap.add_argument("-co", type=float, default=-5.0); ap.add_argument("-pt", type=float, default=50.0)
    ap.add_argument("--convention", choices=["eos", "ess"], default="eos")
    ap.add_argument("--ess-parts", default="detx,hdm,gravity",
                    help="with --convention ess: which ess.amor conventions to switch on (comma list)")
    a = ap.parse_args()

    def files(spec):
        lo, _, hi = spec.partition("-")
        return [os.path.join(a.rp, f"amor{a.Y}n{r:06d}.hdf") for r in range(int(lo), int(hi or lo) + 1)]
    qr = a.q if a.q is not None else (0.005, 0.41)                 # eos default qzRange
    parts = set(a.ess_parts.split(",")) if a.convention == "ess" else set()
    cfg = dict(convention=a.convention, parts=parts, chopper_phase_offset=a.co, low_current=a.pt, y_range=a.y, lambda_range=a.l,
               mu_offset=a.m, theta_range_rel=a.T, q_filter=a.q, q_edges=q_edges(a.r, qr))
    data, info = reduce(files(a.f), files(a.n), cfg)
    hdr = (f"Amor reduction with scipp {sc.__version__} (amor_scipp.py, convention={a.convention} {sorted(parts)})\n"
           f"sample runs {a.f}, direct beam {a.n}; " + ", ".join(f"{k}={v:.6g}" for k, v in info.items()) + "\n"
           "Qz (1/angstrom)  R  sR(eos formula)  sQz  sR(Poisson propagation)")
    np.savetxt(a.o + ".Rqz.ort", data, header=hdr, fmt="%.16e")
    print(a.o, info)


if __name__ == "__main__":
    main()
