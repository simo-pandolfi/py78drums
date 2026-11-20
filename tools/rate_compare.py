#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
P4: the same voices rendered at 48, 96 and 192 kHz (192 kHz = reference).

For every case and rate: output peak, RMS (full band and after decimation to
48 kHz, i.e. below 24 kHz), dominant frequency, time to -40 dB
of the envelope; against 192 kHz: waveform error (both upsampled to 192 kHz,
best lag within +-8 samples at 192 kHz, i.e. the sample-time convention), and
the largest difference of the 1/3-octave band levels from 25 Hz to 20 kHz
(bands within 60 dB of the strongest). Noise cases (seeded, independent
sequences per rate): band levels only (Welch PSD).

Usage (from the py78drums root):
    python tools/rate_compare.py --pywdf <pywdf root> [--out report.json] [--only tonal|noise]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.signal import resample_poly, welch

RATES = (48000, 96000, 192000)
REF = 192000


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--pywdf", required=True)
    p.add_argument("--repo", default=os.path.dirname(here))
    p.add_argument("--out", default=None)
    p.add_argument("--only", choices=("tonal", "noise"), default=None)
    p.add_argument("--match", default=None, help="only cases whose name contains this text")
    return p.parse_args()


def gate(n, fs, pulses, level=4.5):
    v = np.zeros(n)
    for a, g in pulses:
        i = int(round(a * fs))
        v[i:i + int(round(g * fs))] = level
    return v


# -----------------------------------------------------------------------------
# Cases
# -----------------------------------------------------------------------------
def tonal_cases():
    out = []
    for name, p in (("bd", "BDO_PARAMS"), ("cd", "CONGA_PARAMS"), ("hb", "HBONGO_PARAMS"),
                    ("lb", "LBONGO_PARAMS"), ("cl", "CLAVES_PARAMS")):
        out.append((f"tonal_{name}", dict(kind="tonal", params=p, vr4=None, dur=0.5)))
    for vr4 in (0.0, 180e3, 220e3):
        out.append((f"tonal_bd_vr4_{int(vr4 / 1e3)}k", dict(kind="tonal", params="BDO_PARAMS",
                                                             vr4=vr4, dur=0.5)))
    return out


def noise_cases():
    out = []
    for name, voice, g in (("snare", "sd", 0.01), ("lc", "lc", 0.02), ("sc", "sc", 0.02),
                           ("mr", "mr", 0.02)):
        out.append((f"noise_{name}_det", dict(kind="noise", gates={voice: [(0.01, g)]},
                                              noise=False, dur=0.6)))
    out.append(("noise_lc_sd_det", dict(kind="noise", gates={"lc": [(0.01, 0.02)],
                                                             "sd": [(0.25, 0.01)]},
                                        noise=False, dur=0.6)))
    for name, voice, g in (("snare", "sd", 0.01), ("lc", "lc", 0.02)):
        out.append((f"noise_{name}_noise", dict(kind="noise", gates={voice: [(0.01, g)]},
                                                noise=True, dur=0.6)))
    return out


def render(drums, NoiseVoices, cfg, fs, seed=42):
    n = int(round(cfg["dur"] * fs))
    if cfg["kind"] == "tonal":
        d = drums.TonalDrum(fs, **getattr(drums, cfg["params"]))
        if cfg["vr4"] is not None:
            d.set_decay(cfg["vr4"])
        trig = gate(n, fs, [(0.01, 0.01)])
        return np.array([d.tick(trig[i]) for i in range(n)])
    nv = NoiseVoices(fs)
    g = {v: gate(n, fs, cfg["gates"].get(v, [])) for v in ("sd", "lc", "sc", "mr")}
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) if cfg["noise"] else np.zeros(n)
    return np.array([nv.tick(g["sd"][i], g["lc"][i], g["sc"][i], g["mr"][i], noise[i])
                     for i in range(n)])


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def basic(x, fs):
    m = {"peak": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x ** 2)))}
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
    f = np.fft.rfftfreq(8 * len(x), 1 / fs)
    m["f_dom"] = float(f[np.argmax(spec)])
    env = np.abs(x)
    w = max(1, int(0.002 * fs))
    env = np.convolve(env, np.ones(w) / w, mode="same")
    idx = np.nonzero(env > env.max() * 10 ** (-40 / 20))[0]
    m["t40_ms"] = float(1e3 * idx[-1] / fs) if len(idx) else 0.0
    return m


def third_octave_levels(x, fs, psd=False):
    if psd:
        f, p = welch(x, fs, nperseg=min(len(x), int(round(0.17 * fs))))   # same resolution
    else:
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(len(x), 1 / fs)
        p = 2.0 * np.abs(X) ** 2 / (len(x) * fs) / (f[1] - f[0])   # energy density [V^2 s/Hz]
    centers = 1000 * 2.0 ** (np.arange(-16, 14) / 3)          # 25 Hz ... 20 kHz
    lev = []
    for c in centers:
        sel = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        e = np.sum(p[sel]) * (f[1] - f[0])                     # energy (deterministic) or power
        lev.append(10 * np.log10(e + 1e-30))
    return centers, np.array(lev)


def to_ref(x, fs):
    return resample_poly(x, REF // fs, 1) if fs != REF else x


def wave_error(x, fs, ref):
    y = to_ref(x, fs)
    n = min(len(y), len(ref))
    best = None
    for lag in range(-8, 9):
        a = y[max(0, lag):n + min(0, lag)]
        b = ref[max(0, -lag):n - max(0, lag)]
        e = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))
        if best is None or e < best[0]:
            best = (float(e), lag)
    return best


def main():
    a = parse_args()
    sys.path.insert(0, os.path.abspath(a.pywdf))
    sys.path.insert(0, os.path.abspath(a.repo))
    from pe78 import drums
    from pe78.noise_voices import NoiseVoices

    cases = []
    if a.only in (None, "tonal"):
        cases += tonal_cases()
    if a.only in (None, "noise"):
        cases += noise_cases()
    if a.match:
        cases = [c for c in cases if a.match in c[0]]
    report = {}
    for name, cfg in cases:
        t0 = time.time()
        sig = {fs: render(drums, NoiseVoices, cfg, fs) for fs in RATES}
        r = {}
        ref = sig[REF]
        c, lref = third_octave_levels(ref, REF, psd=cfg.get("noise", False))
        keep = lref > lref.max() - 60
        for fs in RATES:
            m = basic(sig[fs], fs)
            x20 = resample_poly(sig[fs], 1, fs // 48000) if fs != 48000 else sig[fs]
            m["rms_48k"] = float(np.sqrt(np.mean(x20 ** 2)))     # audio band (decimated to 48 kHz)
            _, lev = third_octave_levels(sig[fs], fs, psd=cfg.get("noise", False))
            d = lev - lref
            k = keep & (c < 0.45 * fs)
            m["band_max_db"] = float(np.max(np.abs(d[k])))
            m["band_worst_hz"] = float(c[k][np.argmax(np.abs(d[k]))])
            if not cfg.get("noise", False):
                m["wave_err"], m["lag192"] = wave_error(sig[fs], fs, ref)
            r[fs] = m
        report[name] = r
        print(f"{name}  ({time.time() - t0:.0f} s)")
        for fs in RATES:
            m = r[fs]
            we = f"err {100 * m['wave_err']:6.2f} % (lag {m['lag192']:+d})" if "wave_err" in m else ""
            print(f"  {fs // 1000:3d} kHz: peak {m['peak']:.4e} rms {m['rms']:.4e} rms<24k {m['rms_48k']:.4e} "
                  f"f {m['f_dom']:8.1f} Hz t40 {m['t40_ms']:7.1f} ms  bands max "
                  f"{m['band_max_db']:5.2f} dB @ {m['band_worst_hz']:7.0f} Hz  {we}")
        sys.stdout.flush()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
