#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C13 (HB line): the snare output drives the Hi Bongo.

The M253 snare output (pin 5) drives the HB trigger line (pin 6) through D6;
R21 (12 kOhm) is the R14 of the HB edge detector (user, 2026-11-20). Before
C13 the sequencer drove HB with the full 4.5 V gate at every snare step, and
the snare front end saw the line as R21 alone.

Since C13 (decision H-B): the HB voice (TwinTDrum, d6=True) has D6 and gets,
at every sample, the Thevenin (v1_oc, r_th1) of the snare's pin 5 / C22 node
(NoiseVoices.hb_line(), same sample); the snare front end gets the D6
current of the HB voice of the previous sample (tick(..., hb_i_d6)).

1. r_th1 = R_M253 || (R_C22 + R48) (WDF port resistance of C22 at fs), on
   and off; v1_oc = 0 at rest.
2. HB with d6=True and no snare (line=None) equals HB with d6=False
   (D6 reverse-biased or at 0 V).
3. Snare hit (10 ms) against tools/hb_line_ref.py (Fig. 6 and the HB voice in
   one nodal analysis, 192 kHz) at 48 kHz:
   - HB V_n3 within the error of an HB hit alone (same discretisation) x 1.3;
   - HB line (node t1 of the HB voice) within 25 mV;
   - V_C27 within 1 mV outside +-1 ms of the gate edges (with the line seen
     as R21 alone the error is above 20 mV: the HB detector loads C22);
   - the old drive (HB at 4.5 V on snare steps) is at least 2x farther from
     the reference.

Usage (from the py78drums root):
    python -m pytest tests/test_hb_line.py
    python tests/test_hb_line.py          (about 1-2 minutes)

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from pe78 import drums                       # noqa: E402
from pe78.noise_voices import NoiseVoices    # noqa: E402
import hb_line_ref                           # noqa: E402

FS = 48000
DT_REF = 1 / 192000
T_END = 0.15
_REF = {}


def _ref(gates):
    key = repr(gates)
    if key not in _REF:
        _REF[key] = hb_line_ref.simulate(gates, T_END, DT_REF)
    return _REF[key]


def run(gates, mode="d6"):
    """mode: "d6" (C13), "r21" (HB coupled, snare sees R21 only), "old"
    (HB driven at 4.5 V on snare steps). Returns t and signals."""
    nv = NoiseVoices(FS)
    hb = drums.TonalDrum(FS, **drums.HBONGO_PARAMS, d6=(mode != "old"))
    n = int(round(T_END * FS))
    t = np.arange(1, n + 1) / FS
    out = {k: np.empty(n) for k in ("vo", "t1", "c27", "i6")}

    def on(nm, tt):
        return any(a <= tt < a + g for a, g in gates.get(nm, []))
    for k in range(n):
        v_sd = 4.5 if on("sd", t[k]) else 0.0
        v_hb = 4.5 if on("hb", t[k]) else 0.0
        if mode == "old":
            nv.tick(v_sd, 0.0, 0.0, 0.0, 0.0)
            hb.tick(4.5 if (v_sd or v_hb) else 0.0)
        else:
            nv.tick(v_sd, 0.0, 0.0, 0.0, 0.0,
                    hb_i_d6=hb.voice.i_d6 if mode == "d6" else None)
            hb.tick(v_hb, line=nv.hb_line())
            out["i6"][k] = hb.voice.i_d6
        out["vo"][k] = hb.voice.v_out
        out["t1"][k] = hb.voice.root.v_nodes[4]
        out["c27"][k] = nv._V_c27
    return t, out


def _at(t, tr, x):
    j = np.clip(np.round(t / DT_REF).astype(int) - 1, 0, len(tr) - 1)
    return x[j]


def test_thevenin():
    nv = NoiseVoices(FS)
    fe = nv.sd
    r_c22 = 1.0 / (2.0 * fe.C22 * FS)
    for v_sd, r_m in ((0.0, NoiseVoices.R_OFF), (4.5, NoiseVoices.R_ON)):
        nv.tick(v_sd, 0.0, 0.0, 0.0, 0.0)
        ref = 1.0 / (1.0 / r_m + 1.0 / (r_c22 + fe.R48))
        assert abs(nv.hb_line()[1] / ref - 1) < 1e-12, (v_sd, nv.hb_line()[1], ref)
    nv = NoiseVoices(FS)
    nv.tick(0.0, 0.0, 0.0, 0.0, 0.0)
    assert nv.hb_line()[0] == 0.0


def test_idle_line():
    n = FS // 5
    a = drums.TonalDrum(FS, **drums.HBONGO_PARAMS, d6=True)
    b = drums.TonalDrum(FS, **drums.HBONGO_PARAMS)
    d = max(abs(a.tick(4.5 if k < 480 else 0.0) - b.tick(4.5 if k < 480 else 0.0))
            for k in range(n))
    assert d < 1e-6, d


def test_against_reference(verbose=False):
    g_sd = {"sd": [(0.01, 0.01)]}
    g_hb = {"hb": [(0.01, 0.01)]}
    tr, vr = _ref(g_sd)
    trh, vrh = _ref(g_hb)
    t, m = run(g_sd)
    _, mh = run(g_hb)
    _, mr = run(g_sd, "r21")
    _, mo = run(g_sd, "old")
    keep = np.ones(len(t), dtype=bool)
    for e in (0.01, 0.02):
        keep &= np.abs(t - e) > 1e-3

    def rms(x):
        return float(np.sqrt(np.mean(x ** 2)))
    e_vo = m["vo"] - _at(t, tr, vr["n3h"])
    e_hb = mh["vo"] - _at(t, trh, vrh["n3h"])
    e_old = mo["vo"] - _at(t, tr, vr["n3h"])
    e_t1 = np.max(np.abs(m["t1"] - _at(t, tr, vr["h"])))
    e_c27 = np.max(np.abs(m["c27"] - _at(t, tr, vr["c27"]))[keep])
    e_c27_r21 = np.max(np.abs(mr["c27"] - _at(t, tr, vr["c27"]))[keep])
    if verbose:
        print(f"  HB V_n3: snare hit RMS {1e3 * rms(e_vo):.2f} mV (max {np.max(np.abs(e_vo)):.3f} V); "
              f"HB hit alone RMS {1e3 * rms(e_hb):.2f} mV; old 4.5 V drive RMS "
              f"{1e3 * rms(e_old):.2f} mV (max {np.max(np.abs(e_old)):.3f} V)")
        print(f"  HB line: peak {m['t1'].max():.3f} V (ref {vr['h'].max():.3f}), max|d| "
              f"{1e3 * e_t1:.1f} mV; D6 peak {1e3 * m['i6'].max():.3f} mA")
        print(f"  V_C27 (edges excluded): max|d| {1e3 * e_c27:.2f} mV; line as R21 only "
              f"{1e3 * e_c27_r21:.1f} mV; peak {m['c27'].max():.4f} V (ref {vr['c27'].max():.4f})")
    assert rms(e_vo) < 1.3 * rms(e_hb), (rms(e_vo), rms(e_hb))
    assert e_t1 < 25e-3, e_t1
    assert e_c27 < 1e-3, e_c27
    assert e_c27_r21 > 20e-3, e_c27_r21
    assert rms(e_old) > 2 * rms(e_vo), (rms(e_old), rms(e_vo))


if __name__ == "__main__":
    test_thevenin()
    test_idle_line()
    test_against_reference(verbose=True)
    print("OK")
