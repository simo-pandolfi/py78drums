#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
P0 baseline: raw renders of every pe78 voice with a chosen pywdf.

Usage (from the py78drums root):
    python tools/baseline.py [--pywdf <pywdf root>] --out <file.npz>

<pywdf root> is the folder that contains the `pywdf` package (for pywdf-ref:
the repository root; default: the PYWDF_ROOT environment variable, the rule of
pe78/__init__.py). It is put first on sys.path, so it takes precedence over
any pywdf installed with pip; the path actually imported is printed.

Conditions follow bench/: tonal voices 48 kHz, 5 V trigger (wdf_twint_test.py);
snare 48 kHz, 4.5 V trigger from sample 10 (wdf_snare_test.py); cymbals 96 kHz,
20 ms pulses at 50/550/1050 ms (wdf_cymb_test.py). From D11 the triggers
are gates: the M253 level is pe78/m253.py. Added: 100 ms gates, VR4 and
VR9 sweeps. From C15 snare and cymbal cases run NoiseVoices (the other inputs
at 0 V, same names and keys as before) and noise_lc_sd adds a snare hit at
0.25 s on a ringing long cymbal. Outputs are not normalised. Noise is seeded and passed explicitly,
with the level used by drums.py (sigma = NOISE_STD), so it is identical across
runs and pywdf versions.

Writes <file.npz> (signals) and <file>.json (conditions and metrics).
Run each pywdf version in its own process, then compare with
tools/compare_baseline.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--pywdf", default=os.environ.get("PYWDF_ROOT"),
                   help="folder that contains the pywdf package (default: PYWDF_ROOT)")
    p.add_argument("--out", required=True, help="output .npz file")
    p.add_argument("--repo", default=os.path.dirname(here), help="py78drums root (default: parent of tools/)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def metrics(x, fs, dominant=False):
    x = np.asarray(x, dtype=float)
    m = {"finite": bool(np.all(np.isfinite(x)))}
    if not m["finite"]:
        return m
    peak = float(np.max(np.abs(x)))
    m["peak"] = peak
    m["rms"] = float(np.sqrt(np.mean(x ** 2)))
    if peak > 0:
        idx = np.nonzero(np.abs(x) > 0.01 * peak)[0]
        m["t_first_ms"] = 1e3 * idx[0] / fs
        m["t_last_ms"] = 1e3 * idx[-1] / fs     # last sample above 1% of peak
    if dominant and peak > 0:
        y = (x - x.mean()) * np.hanning(len(x))
        spec = np.abs(np.fft.rfft(y))
        f = np.fft.rfftfreq(len(x), 1.0 / fs)
        m["f_dom_hz"] = float(f[1 + np.argmax(spec[1:])])
    return m


# -----------------------------------------------------------------------------
# Cases
# -----------------------------------------------------------------------------
def gate(n, fs, start_s, length_s, level):
    v = np.zeros(n)
    a = int(round(start_s * fs))
    v[a:a + int(round(length_s * fs))] = level
    return v


def tonal_cases(drums):
    fs, dur, level = 48000, 1.0, 5.0
    n = int(dur * fs)
    voices = {"bd": "BDO_PARAMS", "cd": "CONGA_PARAMS", "hb": "HBONGO_PARAMS",
              "lb": "LBONGO_PARAMS", "cl": "CLAVES_PARAMS"}
    cases = []
    for name, pname in voices.items():
        for g_ms in (10, 100):
            cases.append((f"tonal_{name}_gate{g_ms}ms", fs,
                          dict(voice=name, params=pname, gate_ms=g_ms, level=level, vr4=None),
                          n))
    for vr4 in (0.0, 50e3, 150e3, 180e3, 220e3):
        cases.append((f"tonal_bd_vr4_{int(vr4 / 1e3)}k", fs,
                      dict(voice="bd", params="BDO_PARAMS", gate_ms=10, level=level, vr4=vr4), n))
    return cases


def run_tonal(drums, cfg, fs, n):
    d = drums.TonalDrum(fs, **getattr(drums, cfg["params"]))
    if cfg["vr4"] is not None:
        d.set_decay(cfg["vr4"])
    trig = gate(n, fs, 0.0, cfg["gate_ms"] / 1e3, cfg["level"])
    out = np.empty(n)
    for i in range(n):
        out[i] = d.tick(trig[i])
    return {"out": out}


def snare_cases():
    fs, dur = 48000, 0.6
    n = int(dur * fs)
    cases = []
    for g_ms in (10, 100):
        cases.append((f"snare_gate{g_ms}ms", fs, dict(gate_ms=g_ms, vr9u=None), n))
    cases.append(("snare_vr9u_110k", fs, dict(gate_ms=10, vr9u=110e3), n))
    return cases


def run_snare(drums, cfg, fs, n, noise):
    # C15: the snare is part of NoiseVoices (cymbal and maracas inputs at 0 V);
    # keys kept from the old SnareDrum cases (V_base = TR3 base, V_col = TR3
    # collector) so that compare_baseline.py can compare across C15.
    d = drums.NoiseVoices(fs)
    if cfg["vr9u"] is not None:
        d.set_vr9(cfg["vr9u"] / d.VR9)
    trig = gate(n, fs, 10.0 / fs, cfg["gate_ms"] / 1e3, 4.5)
    keys = (("V_c27", "_V_c27"), ("V_base", "_V_b3"), ("V_col", "_V_col3"))
    out = np.empty(n)
    st = {k: np.empty(n) for k, _ in keys}
    for i in range(n):
        out[i] = d.tick(trig[i], 0.0, 0.0, 0.0, noise=noise[i])
        for k, attr in keys:
            st[k][i] = getattr(d, attr)
    return {"out": out, **st}


def cymb_cases():
    fs = 96000
    cases = []
    for g_ms in (20, 100):
        n = int(2.0 * fs)
        cases.append((f"cymb_pattern_gate{g_ms}ms", fs, dict(gate_ms=g_ms, vr9l=None, voices="lc,sc,mr"), n))
    for vr9l in (100e3, 195e3):
        n = int(0.6 * fs)
        cases.append((f"cymb_lc_vr9l_{int(vr9l / 1e3)}k", fs, dict(gate_ms=20, vr9l=vr9l, voices="lc"), n))
    # C15: snare hit while the long cymbal rings (shared noise node and VR9)
    cases.append(("noise_lc_sd", fs, dict(gate_ms=20, vr9l=None, voices="lc", sd_s=0.25), int(0.6 * fs)))
    return cases


def run_cymb(drums, cfg, fs, n, noise):
    # C15: cymbals and maracas are part of NoiseVoices (snare input at 0 V
    # unless cfg["sd_s"] is set); keys kept from the old CymbDrum cases
    # (V_col = TR4 collector).
    d = drums.NoiseVoices(fs)
    if cfg["vr9l"] is not None:
        d.set_vr9(1.0 - cfg["vr9l"] / d.VR9)
    g = cfg["gate_ms"] / 1e3
    voices = cfg["voices"].split(",")
    lc = gate(n, fs, 0.05, g, 4.5)
    sc = gate(n, fs, 0.55, g, 4.5) if "sc" in voices else np.zeros(n)
    mr = gate(n, fs, 1.05, g, 4.5) if "mr" in voices else np.zeros(n)
    sd = gate(n, fs, cfg["sd_s"], 0.01, 4.5) if cfg.get("sd_s") is not None else np.zeros(n)
    keys = (("V_env", "_V_env"), ("V_c23", "_V_c23"), ("V_c24", "_V_c24"),
            ("V_c26", "_V_c26"), ("V_col", "_V_col4"))
    out = np.empty(n)
    st = {k: np.empty(n) for k, _ in keys}
    for i in range(n):
        out[i] = d.tick(sd[i], lc[i], sc[i], mr[i], noise=noise[i])
        for k, attr in keys:
            st[k][i] = getattr(d, attr)
    return {"out": out, **st}


# -----------------------------------------------------------------------------
def main():
    a = parse_args()
    if not a.pywdf:
        sys.exit("error: give --pywdf or set PYWDF_ROOT")
    os.environ["PYWDF_ROOT"] = os.path.abspath(a.pywdf)     # read by pe78/__init__.py
    sys.path.insert(0, os.path.abspath(a.repo))
    sys.path.insert(0, os.path.abspath(a.pywdf))
    import pywdf
    from pe78 import drums
    pywdf_file = os.path.abspath(pywdf.__file__)
    print(f"pywdf : {pywdf_file}")
    print(f"pe78  : {os.path.abspath(drums.__file__)}")
    if not pywdf_file.startswith(os.path.abspath(a.pywdf)):
        sys.exit("error: the imported pywdf is not the one given with --pywdf")

    signals, report = {}, {"pywdf": pywdf_file, "seed": a.seed,
                           "noise_std": float(drums.NOISE_STD), "numpy": np.__version__,
                           "cases": {}}

    def store(name, fs, cfg, data, dominant):
        for k, v in data.items():
            signals[f"{name}/{k}"] = v
        report["cases"][name] = {"fs": fs, "cfg": cfg,
                                 "metrics": {k: metrics(v, fs, dominant and k == "out")
                                             for k, v in data.items()}}

    t0 = time.perf_counter()
    for name, fs, cfg, n in tonal_cases(drums):
        store(name, fs, cfg, run_tonal(drums, cfg, fs, n), True)
        print(f"  {name:28s} done  ({time.perf_counter() - t0:6.1f} s)")
    for name, fs, cfg, n in snare_cases():
        noise = np.random.default_rng(a.seed).normal(0.0, drums.NOISE_STD, n)
        store(name, fs, cfg, run_snare(drums, cfg, fs, n, noise), False)
        print(f"  {name:28s} done  ({time.perf_counter() - t0:6.1f} s)")
    for name, fs, cfg, n in cymb_cases():
        noise = np.random.default_rng(a.seed).normal(0.0, drums.NOISE_STD, n)
        store(name, fs, cfg, run_cymb(drums, cfg, fs, n, noise), False)
        print(f"  {name:28s} done  ({time.perf_counter() - t0:6.1f} s)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **signals)
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1)
    print(f"written {a.out} and {os.path.splitext(a.out)[0]}.json")


if __name__ == "__main__":
    main()
