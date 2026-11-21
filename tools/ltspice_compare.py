#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
P4: NoiseVoices against the LTspice simulation of Fig. 6 (same gates).

LTspice reference: ltspice/Practical-Electronics-1978-01_Pagina_24_Fig6.asc
with `.tran 0 1.6 0 0.25u`, gates of 15 ms at 5.2 V through D1-D4: SD 0.01 s,
LC 0.15 s, SC 1.1 s, MR 1.4 s. Two runs: with B1 and with B1 = 0. For each run
(all files optional, looked up in --ltdir):
    noise_drum.wav, noise_drum_noB1.wav      .wave V(out), 16 bit 48 kHz, +-1 V
    *Fig6_envelope.txt, *Fig6_noB1_envelope.txt   V(envelope), TR4 base
    *Fig6_n008.txt, *Fig6_noB1_n008.txt           V(n008), TR3 base
    *Fig6_out.txt, *Fig6_noB1_out.txt             V(out) (optional, preferred)
Text exports: LTspice "Export data as text" (time, value; tab separated).

Model: pe78.noise_voices.NoiseVoices at 96 kHz (--fs), the same gates (M253
inputs), B1 = 0 run with noise = 0, B1 run with --seeds unit-variance
sequences (numpy default_rng(seed)). Model sample k is at t = (k + 1) / fs
(tests/_nv_common.py); LTspice traces are resampled to 4.8 MHz on a uniform
grid, then to fs (resample_poly).

Output of the B1 run: the .wave file is point-sampled (no anti-alias filter)
and the collector noise extends to about 1 MHz: the WAV carries the aliases
of 24 kHz ... 1 MHz. The in-band LTspice output is the V(out) text export
(decimated to fs); without it, it is rebuilt from the exported base
voltages: Ic = IS*(exp(Vbe/(NF*VT)) - 1) (BC547B card: IS, NF; VT at
27 C), collector network of the .asc (L1, L2
with Rser 61.662 Ohm and Rpar 9.42 MOhm, R53, R54, C30, VR9U/VR9L, R55, C31,
RLoad) at 4.8 MHz, bilinear, then decimated. The rebuild is checked against
the B1 = 0 WAV sample by sample, and its point-sampled version against the
B1 WAV level by level (aliasing check).

Measures
  B1 = 0: output (WAV, 48 kHz, model point-sampled at the WAV times; and the
          V(out) export at fs) sample by sample per segment; V(envelope) and
          V(n008) sample by sample; times of ENV > 0.5 V; peaks.
  Rebuild: rebuilt output against the V(out) exports, when present.
  B1:     output levels per segment (RMS), 5 ms RMS envelopes, 1/3-octave
          levels (Welch, model averaged over seeds); V(envelope) and V(n008):
          slow part (5 ms mean) and noise (B1 run minus B1 = 0 run) in bands.

Diagnostics (not the model; used to weigh the calibration options):
  --m253 V_ON,R_ON,R_OFF   override the M253 output of NoiseVoices
  --post RSER,GAIN         output recomputed from the model Ic3, Ic4 with
                           the collector network above, inductor Rser [Ohm]
                           and collector currents x GAIN (the model's own
                           Rser, no Rpar, GAIN 1 = the model output; checked
                           at start)

Usage (from the py78drums root):
    python tools/ltspice_compare.py [--pywdf <pywdf root>] --ltdir <folder>
        [--seeds 4] [--fs 96000] [--m253 4.85,350,1e12] [--post 61.662,1.61]
        [--out report.json]
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.signal import bilinear_zpk, resample_poly, sosfilt, ss2zpk, welch, zpk2sos

GATES = {"sd": 0.01, "lc": 0.15, "sc": 1.1, "mr": 1.4}
GATE_LEN = 0.015
T_END = 1.6
SEGS = (("SD", 0.0, 0.15), ("LC", 0.15, 1.1), ("SC", 1.1, 1.4), ("MR", 1.4, 1.6))
FS_HI = 4_800_000             # LTspice traces resampled here (max step 0.25 us; 100 x 48 kHz)
# BC547B card (LTspice standard.bjt): IS, NF; VT at 27 C
IS_BJT, NF_BJT, VT_27C = 2.39e-14, 1.008, 25.852e-3
# .asc collector network
L1 = L2 = 100e-3
L_RSER, L_RPAR = 61.662, 9.42e6
R53 = R54 = 4700.0
C30, R55, C31, RLOAD = 0.02e-6, 22e3, 0.1e-6, 10e3
VR9U, VR9L = 25e3, 195e3
THIRD = 1000 * 2.0 ** (np.arange(-16, 14) / 3)       # 25 Hz ... 20 kHz


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--pywdf", default=os.environ.get("PYWDF_ROOT"),
                   help="folder that contains the pywdf package (default: PYWDF_ROOT)")
    p.add_argument("--repo", default=os.path.dirname(here))
    p.add_argument("--ltdir", required=True)
    p.add_argument("--fs", type=int, default=96000)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--m253", default=None, help="V_ON,R_ON,R_OFF")
    p.add_argument("--post", default=None, help="RSER,GAIN")
    p.add_argument("--out", default=None)
    return p.parse_args()


# -----------------------------------------------------------------------------
# LTspice data
# -----------------------------------------------------------------------------
def find(ltdir, pattern, exclude=None):
    hits = [f for f in glob.glob(os.path.join(ltdir, pattern))
            if exclude is None or exclude not in os.path.basename(f)]
    if len(hits) > 1:
        raise SystemExit(f"more than one file for {pattern}: {hits}")
    return hits[0] if hits else None


def load_trace(path):
    """Text export -> values on the uniform FS_HI grid from 0 to T_END."""
    a = np.loadtxt(path, skiprows=1, dtype=np.float64)
    t = np.arange(int(round(T_END * FS_HI)) + 1) / FS_HI
    return np.interp(t, a[:, 0], a[:, 1]), a


def load_wav(path):
    fs, w = wavfile.read(path)
    if w.dtype != np.int16:
        raise SystemExit(f"{path}: expected 16 bit PCM")
    return fs, w.astype(np.float64) / 32768.0     # .wave: +-1 V full scale


def hi_to(x, fs):
    """FS_HI -> fs (48, 96 or 192 kHz)."""
    g = np.gcd(FS_HI, fs)
    return resample_poly(x, fs // g, FS_HI // g)


def lt_files(ltdir):
    f = {"wav_b1": find(ltdir, "noise_drum.wav"), "wav_nob1": find(ltdir, "noise_drum_noB1.wav")}
    for key, name in (("env", "envelope"), ("n008", "n008"), ("out", "out")):
        f[f"{key}_b1"] = find(ltdir, f"*Fig6_{name}.txt")
        f[f"{key}_nob1"] = find(ltdir, f"*Fig6_noB1_{name}.txt")
    return f


# -----------------------------------------------------------------------------
# Collector network of the .asc (rebuild and --post)
# -----------------------------------------------------------------------------
def collector_sos(fs, rser=L_RSER, rpar=L_RPAR):
    """SOS filters Ic3 -> V(out) and Ic4 -> V(out) (Ic into the collector).

    States: iL1, iL2 (from +12 V into the collector), V(c3) on C30, V(C31).
    Algebraic nodes: c4, wiper W. Bilinear at fs."""
    gp = 0.0 if rpar is None else 1.0 / rpar
    ro = R55 + RLOAD
    M = np.array([[1 / R54 + gp + 1 / VR9L, -1 / VR9L],
                  [-1 / VR9L, 1 / VR9U + 1 / VR9L + 1 / ro]])
    Mi = np.linalg.inv(M)
    Yx = Mi @ np.array([[0, 1, 0, 0], [0, 0, 1 / VR9U, 1 / ro]])
    Yu = Mi @ np.array([[0, -1], [0, 0]])
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    A[0, 2], A[0, 0] = -1 / L1, -rser / L1
    A[1, :] = -Yx[0] / L2
    A[1, 1] += -rser / L2
    B[1, :] = -Yu[0] / L2
    A[2, 0] = 1 / C30
    A[2, 2] = (-1 / R53 - gp - 1 / VR9U) / C30
    A[2, :] += Yx[1] / VR9U / C30
    B[2, :] = Yu[1] / VR9U / C30
    B[2, 0] += -1 / C30
    A[3, :] = Yx[1] / ro / C31
    A[3, 3] += -1 / ro / C31
    B[3, :] = Yu[1] / ro / C31
    C = (Yx[1] - np.array([0, 0, 0, 1])) * RLOAD / ro
    D = Yu[1] * RLOAD / ro
    sos = []
    for j in range(2):
        # zeros and poles in s (well conditioned), then bilinear: the zero
        # of C31 at DC stays at z = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # BadCoefficients from ss2tf inside
            z, p, k = ss2zpk(A, B[:, j:j + 1], C[None, :], D[None, j:j + 1])
        z = np.where(np.abs(z) < 1e-6, 0.0, z)
        sos.append(zpk2sos(*bilinear_zpk(z, p, k, fs)))
    return sos


def collector_out(ic3, ic4, fs, rser=L_RSER, rpar=L_RPAR):
    s3, s4 = collector_sos(fs, rser, rpar)
    return sosfilt(s3, ic3) + sosfilt(s4, ic4)


def ic_bjt(vbe):
    return IS_BJT * np.expm1(vbe / (NF_BJT * VT_27C))


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
REC = {"out": "_V_out", "env": "_V_env", "b3": "_V_b3", "ic3": "_Ic3", "ic4": "_Ic4"}


def render(NoiseVoices, fs, noise, m253=None):
    cls = NoiseVoices
    if m253 is not None:
        cls = type("NoiseVoicesM253", (NoiseVoices,), {})
        cls.V_ON, cls.R_ON, cls.R_OFF = m253
    nv = cls(fs)
    n = int(round(T_END * fs))
    t = np.arange(1, n + 1) / fs
    g = {k: np.where((t >= a) & (t < a + GATE_LEN), nv.V_ON, 0.0) for k, a in GATES.items()}
    out = {k: np.zeros(n) for k in REC}
    for i in range(n):
        nv.tick(g["sd"][i], g["lc"][i], g["sc"][i], g["mr"][i], noise[i])
        for k, attr in REC.items():
            out[k][i] = getattr(nv, attr)
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def seg_mask(t, a, b):
    return (t >= a) & (t < b)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def db(a, b):
    return float(20 * np.log10(a / b)) if a > 0 and b > 0 else float("nan")


def last_above(t, v, a, b, thr=0.5):
    s = seg_mask(t, a, b)
    i = np.nonzero(v[s] > thr)[0]
    return float(t[s][i[-1]]) if len(i) else float("nan")


def env_db(x, fs, win=0.005):
    """RMS envelope in non-overlapping windows [dB re 1 V]."""
    w = int(round(win * fs))
    n = len(x) // w
    e = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1))
    return (np.arange(n) + 0.5) * w / fs, 20 * np.log10(e + 1e-15)


def third_octave(x, fs):
    nper = min(len(x), 8192)
    f, p = welch(x, fs, nperseg=nper)
    df = f[1] - f[0]
    lev = []
    for c in THIRD:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        lev.append(10 * np.log10(np.sum(p[sel]) * df + 1e-30) if np.any(sel) else np.nan)
    return np.array(lev), nper / fs


def band_density(x, fs, bands):
    f, p = welch(x, fs, nperseg=min(len(x), 4096))
    return [float(np.mean(p[(f >= lo) & (f < hi)])) for lo, hi in bands]


def out_table(title, t, ref, y):
    """Per segment: RMS, level difference, RMS error / RMS ref, max error, peaks."""
    rows = {}
    print(f"{title:22s} LT rms      model rms   diff dB  err rms/LT  max|err|   LT peak  model peak")
    for nm, a0, b0 in SEGS:
        s = seg_mask(t, a0, b0)
        rows[nm] = dict(lt_rms=rms(ref[s]), model_rms=rms(y[s]), diff_db=db(rms(y[s]), rms(ref[s])),
                        err_rel=rms(y[s] - ref[s]) / rms(ref[s]),
                        err_max=float(np.max(np.abs(y[s] - ref[s]))),
                        lt_peak=float(np.max(np.abs(ref[s]))), model_peak=float(np.max(np.abs(y[s]))))
        q = rows[nm]
        print(f"  {nm}   {q['lt_rms']:.3e}  {q['model_rms']:.3e}  {q['diff_db']:+6.2f}  {q['err_rel']:8.3f}"
              f"  {q['err_max']:.2e}  {q['lt_peak']:.2e}  {q['model_peak']:.2e}")
    return rows


# -----------------------------------------------------------------------------
def main():
    a = parse_args()
    if not a.pywdf:
        sys.exit("error: give --pywdf or set PYWDF_ROOT")
    os.environ["PYWDF_ROOT"] = os.path.abspath(a.pywdf)     # read by pe78/__init__.py
    sys.path.insert(0, os.path.abspath(a.pywdf))
    sys.path.insert(0, os.path.abspath(a.repo))
    import pywdf
    from pe78.noise_voices import NoiseVoices
    print("pywdf:", os.path.dirname(pywdf.__file__))
    fs = a.fs
    m253 = tuple(float(v) for v in a.m253.split(",")) if a.m253 else None
    post = tuple(float(v) for v in a.post.split(",")) if a.post else None
    files = lt_files(a.ltdir)
    for k, v in files.items():
        print(f"  {k:9s} {v}")
    rep = {"fs": fs, "seeds": a.seeds, "m253": m253, "post": post, "files": files}

    # ---- check of the collector network against the model (its own Rser, GAIN 1)
    rser_m = float(getattr(NoiseVoices, "L_RSER", 0.0))
    nv0 = render(NoiseVoices, fs, np.zeros(int(round(T_END * fs))), m253)
    chk = collector_out(nv0["ic3"], nv0["ic4"], fs, rser=rser_m, rpar=None)
    e = float(np.max(np.abs(chk - nv0["out"])))
    print(f"collector network (Rser {rser_m:g} Ohm, no Rpar) vs model output: "
          f"max {e:.2e} V of {np.max(np.abs(nv0['out'])):.2e}")
    rep["collector_check"] = e

    def model_out(r):
        if post is None:
            return r["out"]
        return collector_out(post[1] * r["ic3"], post[1] * r["ic4"], fs, rser=post[0], rpar=L_RPAR)

    n = int(round(T_END * fs))
    tm = np.arange(1, n + 1) / fs                 # model sample times
    lt = {}                                       # LTspice at fs, aligned to tm
    raw = {}
    for run in ("nob1", "b1"):
        for key in ("env", "n008", "out"):
            p = files[f"{key}_{run}"]
            if p:
                hi, _ = load_trace(p)
                raw[f"{key}_{run}"] = hi
                lt[f"{key}_{run}"] = hi_to(hi, fs)[1:n + 1]

    # ---- rebuilt LTspice output (from the base traces)
    for run in ("nob1", "b1"):
        if f"env_{run}" in raw and f"n008_{run}" in raw:
            y = collector_out(ic_bjt(raw[f"n008_{run}"]), ic_bjt(raw[f"env_{run}"]), FS_HI)
            raw[f"rebuilt_{run}"] = y
            lt[f"rebuilt_{run}"] = hi_to(y, fs)[1:n + 1]

    # ===================== B1 = 0 =====================
    print("\n=== B1 = 0: model sample by sample ===")
    r0 = {}
    if files["wav_nob1"]:
        fw, w = load_wav(files["wav_nob1"])
        tw = np.arange(len(w)) / fw
        step = fs // fw
        mo = model_out(nv0)
        y = mo[step - 1::step][:len(w)]           # model at t = k/fw (k >= 1)
        w1, tw1 = w[1:len(y) + 1], tw[1:len(y) + 1]
        y = y[:len(w1)]
        rows = out_table("output (WAV 48 kHz)", tw1, w1, y)
        r0["out"] = rows
        if "rebuilt_nob1" in raw:
            yb = raw["rebuilt_nob1"][::FS_HI // fw][:len(w)]      # FS_HI = 100 x 48 kHz
            chk = {}
            print("rebuild check (LTspice bases -> Ic -> collectors, point-sampled) vs WAV:")
            for nm, a0, b0 in SEGS:
                s = seg_mask(tw, a0, b0)
                chk[nm] = dict(diff_db=db(rms(yb[s]), rms(w[s])), err_rel=rms(yb[s] - w[s]) / rms(w[s]))
                print(f"  {nm}   level {chk[nm]['diff_db']:+6.2f} dB  err rms/LT {chk[nm]['err_rel']:.3f}")
            r0["rebuild_check"] = chk
    mo = model_out(nv0)
    if "out_nob1" in lt:
        r0["out_export"] = out_table(f"output (V(out), {fs // 1000} kHz)", tm, lt["out_nob1"], mo)
    for run in ("nob1", "b1"):
        if f"out_{run}" in lt and f"rebuilt_{run}" in lt:
            chk = {}
            print(f"rebuild check, {'B1 = 0' if run == 'nob1' else 'B1'}: rebuilt vs V(out) export, {fs // 1000} kHz")
            ref, yb = lt[f"out_{run}"], lt[f"rebuilt_{run}"]
            for nm, a0, b0 in SEGS:
                s = seg_mask(tm, a0, b0)
                chk[nm] = dict(diff_db=db(rms(yb[s]), rms(ref[s])), err_rel=rms(yb[s] - ref[s]) / rms(ref[s]))
                print(f"  {nm}   level {chk[nm]['diff_db']:+6.2f} dB  err rms/LT {chk[nm]['err_rel']:.3f}")
            rep.setdefault("rebuild_vs_export", {})[run] = chk
    for key, mk in (("env", "env"), ("n008", "b3")):
        if f"{key}_nob1" not in lt:
            continue
        v_lt, v_m = lt[f"{key}_nob1"], nv0[mk]
        rows = {}
        print(f"V({key})           LT peak   model peak   err max    err rms   LT >0.5 V until  model")
        for nm, a0, b0 in SEGS:
            s = seg_mask(tm, a0, b0)
            rows[nm] = dict(lt_peak=float(v_lt[s].max()), model_peak=float(v_m[s].max()),
                            err_max=float(np.max(np.abs(v_m[s] - v_lt[s]))), err_rms=rms(v_m[s] - v_lt[s]),
                            lt_t05=last_above(tm, v_lt, a0, b0), model_t05=last_above(tm, v_m, a0, b0))
            q = rows[nm]
            print(f"  {nm}   {q['lt_peak']:9.4f}  {q['model_peak']:9.4f}  {q['err_max']:.2e}  {q['err_rms']:.2e}"
                  f"   {q['lt_t05']:8.4f}  {q['model_t05']:8.4f}")
        r0[key] = rows
    rep["nob1"] = r0

    # ===================== B1 =====================
    print(f"\n=== B1: model with {a.seeds} noise seeds ===")
    runs = []
    for seed in range(a.seeds):
        t0 = time.time()
        runs.append(render(NoiseVoices, fs, np.random.default_rng(seed).standard_normal(n), m253))
        print(f"  seed {seed}: {time.time() - t0:.0f} s")
        sys.stdout.flush()
    mouts = [model_out(r) for r in runs]
    r1 = {}
    ref_key = "out_b1" if "out_b1" in lt else ("rebuilt_b1" if "rebuilt_b1" in lt else None)
    if ref_key:
        print(f"LTspice output: {'V(out) text export' if ref_key == 'out_b1' else 'rebuilt from the bases'}")
        y_lt = lt[ref_key]
        if files["wav_b1"]:
            fw, w = load_wav(files["wav_b1"])
            tw = np.arange(len(w)) / fw
            src = raw["out_b1"] if ref_key == "out_b1" else raw["rebuilt_b1"]
            yp = src[::FS_HI // fw][:len(w)]
            yf = hi_to(src, fw)[:len(w)]
            al = {}
            print("aliasing of the .wave:    WAV rms   point-sampled  in band (<24 kHz)   WAV/in band dB")
            for nm, a0, b0 in SEGS:
                s = seg_mask(tw, a0, b0)
                al[nm] = dict(wav=rms(w[s]), point=rms(yp[s]), inband=rms(yf[s]), wav_over_inband_db=db(rms(w[s]), rms(yf[s])))
                print(f"  {nm}   {al[nm]['wav']:.3e}  {al[nm]['point']:.3e}  {al[nm]['inband']:.3e}  {al[nm]['wav_over_inband_db']:+6.2f}")
            r1["wav_aliasing"] = al
        lev = {}
        print("output RMS            LT          model (mean of seeds, min..max)   diff dB")
        for nm, a0, b0 in SEGS:
            s = seg_mask(tm, a0, b0)
            mr = [rms(m[s]) for m in mouts]
            lev[nm] = dict(lt=rms(y_lt[s]), model=float(np.sqrt(np.mean(np.square(mr)))),
                           model_min=min(mr), model_max=max(mr))
            lev[nm]["diff_db"] = db(lev[nm]["model"], lev[nm]["lt"])
            q = lev[nm]
            print(f"  {nm}   {q['lt']:.3e}   {q['model']:.3e} ({q['model_min']:.2e}..{q['model_max']:.2e})  {q['diff_db']:+6.2f}")
        r1["levels"] = lev
        # envelopes (5 ms RMS, model: power mean over seeds)
        te, e_lt = env_db(y_lt, fs)
        e_m = 10 * np.log10(np.mean([10 ** (env_db(m, fs)[1] / 10) for m in mouts], axis=0))
        envr = {}
        print("envelope (5 ms RMS)   peak LT dBV  peak model   diff dB median  max|diff| (LT > peak-40 dB)  -40 dB end LT  model")
        for nm, a0, b0 in SEGS:
            s = seg_mask(te, a0, b0)
            pk = e_lt[s].max()
            k = s & (e_lt > pk - 40)
            d = e_m[k] - e_lt[k]
            end_lt = float(te[k][-1])
            km = s & (e_m > e_m[s].max() - 40)
            end_m = float(te[km][-1])
            envr[nm] = dict(peak_lt=float(pk), peak_model=float(e_m[s].max()), diff_median=float(np.median(d)),
                            diff_maxabs=float(np.max(np.abs(d))), end40_lt=end_lt, end40_model=end_m)
            q = envr[nm]
            print(f"  {nm}   {q['peak_lt']:7.1f}   {q['peak_model']:7.1f}   {q['diff_median']:+6.2f}   {q['diff_maxabs']:6.2f}"
                  f"   {q['end40_lt']:.3f}  {q['end40_model']:.3f}")
        r1["envelopes"] = envr
        # third-octave levels
        tho = {}
        print("1/3 octave, model - LT [dB] (bands with >= 4 Welch resolution cells, within 50 dB of the max)")
        print("        " + " ".join(f"{c:>6.0f}" for c in THIRD[::3]))
        for nm, a0, b0 in SEGS:
            s = seg_mask(tm, a0, b0)
            l_lt, tres = third_octave(y_lt[s], fs)
            l_m = 10 * np.log10(np.mean([10 ** (third_octave(m[s], fs)[0] / 10) for m in mouts], axis=0))
            ok = (THIRD * 2 ** (1 / 6) - THIRD / 2 ** (1 / 6) >= 4 / tres) & (l_lt > np.nanmax(l_lt) - 50)
            d = np.where(ok, l_m - l_lt, np.nan)
            tho[nm] = dict(centers=THIRD.tolist(), lt=l_lt.tolist(), model=l_m.tolist(), diff=d.tolist())
            print(f"  {nm}   " + " ".join(f"{v:+6.1f}" if np.isfinite(v) else "     -" for v in d[::3]))
        r1["third_octave"] = tho

    # bases: slow part and noise
    bands = ((50, 200), (200, 1000), (1000, 5000), (5000, 20000))
    for key, mk in (("env", "env"), ("n008", "b3")):
        if f"{key}_b1" not in lt:
            continue
        rows = {}
        w5 = int(round(0.005 * fs))
        slow_lt = np.convolve(lt[f"{key}_b1"], np.ones(w5) / w5, mode="same")
        slow_m = np.mean([np.convolve(r[mk], np.ones(w5) / w5, mode="same") for r in runs], axis=0)
        have0 = f"{key}_nob1" in lt
        print(f"V({key}) B1:  slow part (5 ms mean) err max / rms;  LT >0.5 V until / model;"
              + ("  noise (B1 - no B1) model - LT [dB] " + " ".join(f"{lo}-{hi}" for lo, hi in bands) if have0 else ""))
        for nm, a0, b0 in SEGS:
            s = seg_mask(tm, a0, b0)
            # 5 ms centred mean: the next gate leaks into the last 2.5 ms of a segment
            q = dict(slow_err_max=float(np.max(np.abs(slow_m[s] - slow_lt[s]))), slow_err_rms=rms(slow_m[s] - slow_lt[s]),
                     lt_t05=last_above(tm, slow_lt, a0, b0 - 0.005), model_t05=last_above(tm, slow_m, a0, b0 - 0.005))
            line = f"  {nm}   {q['slow_err_max']:.2e} / {q['slow_err_rms']:.2e}   {q['lt_t05']:.4f} / {q['model_t05']:.4f}"
            if have0:
                d_lt = band_density((lt[f"{key}_b1"] - lt[f"{key}_nob1"])[s], fs, bands)
                d_m = np.mean([band_density((r[mk] - nv0[mk])[s], fs, bands) for r in runs], axis=0)
                q["noise_diff_db"] = [float(10 * np.log10(m_ / l_)) for m_, l_ in zip(d_m, d_lt)]
                q["noise_lt_vrthz"] = [float(np.sqrt(v)) for v in d_lt]
                line += "     " + " ".join(f"{v:+6.1f}" for v in q["noise_diff_db"])
            rows[nm] = q
            print(line)
        r1[key] = rows
    rep["b1"] = r1
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
