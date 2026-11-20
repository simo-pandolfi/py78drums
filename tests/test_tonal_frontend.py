#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C13 and C14: edge detector of the tonal voices.

C14: D2 was wdf.Diode(Is = 2.52 nA) with the default n_diodes = 1, i.e.
emission coefficient N = 1 (drop about 0.33 V at 1 mA). The LTspice
standard.dio 1N4148 (used for D6-D9 since C6/C8) has N = 1.752: drops
0.38/0.48/0.58 V at 0.01/0.1/1 mA (user's data).
C13: D2 went to GND and the twin-T was fed by a source equal to the voltage
on R13 behind another 27 kOhm (R13 counted twice, no coupling back); R14 was
shorted by a 1 mOhm source. In Fig. 7 the cathode of D2 is on the C6/C7/R16
node: since C13 the detector and the twin-T are one circuit (TwinTDrum) with
D2 and the gate at one Newton root; the M253 output is the switched source of
noise_voices.py (on: V behind 350 Ohm, off: 0 V behind 10 MOhm).

1. D2 law: N = 1.752, drops at 0.01/0.1/1 mA (+-10 mV); at every sample of a
   hit the D2 port current equals the Shockley law of its port voltage.
2. Topology: R13 is one leaf (27 kOhm) between t2 and the anode of D2; D2 is
   the port anode-n1; R14 is a leaf on t1; the M253 leaf switches between
   350 Ohm and 10 MOhm.
3. Whole voice against tools/tonal_voice_ref.py (nodal, trapezoidal, Newton
   on all nodes) at the same sample rate: V_n3 within 1e-8 V and the D2
   current within 1e-12 A, for BD (VR4 150 and 470 kOhm, 10 and 100 ms gates,
   retrigger), HB (VR4 0), CL (VR4 150 kOhm).
4. D2 conducts on the rising edge of the trigger (peak 0.12-0.14 mA) and, in
   self-oscillation (BD, VR4 470 kOhm), on the negative swings of n1; after a
   1 mV step on C7 (small signal) it does not conduct.

Usage (from the py78drums root):
    python -m pytest tests/test_tonal_frontend.py
    python tests/test_tonal_frontend.py      (about 1 minute)

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78 import drums                    # noqa: E402
from pe78.twint import TwinTDrum          # noqa: E402
import tonal_voice_ref as ref             # noqa: E402

FS = 48000
TD = TwinTDrum


def drop(i, n=TD.D_N):
    return n * TD.VT_27C * np.log1p(i / TD.D_IS)


def gate_fn(*pulses, level=4.5):
    """pulses: (start_s, length_s); returns gate(k)."""
    spans = [(int(round(a * FS)), int(round((a + g) * FS))) for a, g in pulses]
    return lambda k: level if any(a <= k < b for a, b in spans) else 0.0


def run_model(name, n, gate, vr4=None):
    d = drums.TonalDrum(FS, **getattr(drums, name))
    if vr4 is not None:
        d.set_decay(vr4)
    vo, i2, v2 = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n):
        d.tick(gate(k))
        vo[k], i2[k] = d.voice.v_out, d.voice.i_d2
        v2[k] = d.voice.root.v_ports[0]
    return vo, i2, v2, d.voice


def test_d2_law():
    for i, r in ((1e-5, 0.38), (1e-4, 0.48), (1e-3, 0.58)):
        assert abs(drop(i) - r) < 0.01, (i, drop(i))
    assert drop(1e-3) - drop(1e-3, n=1.0) > 0.24          # N = 1 before C14
    _, i2, v2, _ = run_model("BDO_PARAMS", FS // 4, gate_fn((0.01, 0.01)))
    law = TD.D_IS * (np.exp(v2 / (TD.D_N * TD.VT_27C)) - 1.0)
    assert np.max(np.abs(i2 - law)) < 1e-15, np.max(np.abs(i2 - law))
    on = i2 > 1e-7
    assert on.sum() > 10
    assert np.max(np.abs(v2[on] - drop(i2[on]))) < 1e-9


def test_topology():
    v = drums.TonalDrum(FS, **drums.BDO_PARAMS).voice
    leaves = {id(leaf): (p, q) for leaf, p, q in v.root.leaves}
    r27 = [leaf for leaf, _, _ in v.root.leaves if leaf.Rp == 27e3]
    assert r27 == [v.r13] and leaves[id(v.r13)] == (5, 6)         # t2 - a, once
    assert v.root.AN[:, 0].tolist() == [0, -1, 0, 0, 0, 0, 1]      # D2: a -> n1
    assert leaves[id(v.r14)] == (4, -1) and v.r14.Rp == 12e3
    assert leaves[id(v.m253)] == (4, -1) and v.m253.Rp == TD.R_OFF
    v.process_sample(4.5)
    assert v.m253.Rp == TD.R_ON
    v.process_sample(0.0)
    assert v.m253.Rp == TD.R_OFF


def test_against_reference(verbose=False):
    n = FS // 2
    cases = [("BDO_PARAMS", 150e3, gate_fn((0.01, 0.01)), "10 ms"),
             ("BDO_PARAMS", 150e3, gate_fn((0.01, 0.1)), "100 ms"),
             ("BDO_PARAMS", 150e3, gate_fn((0.01, 0.01), (0.06, 0.01)), "retrigger"),
             ("BDO_PARAMS", 470e3, gate_fn((0.01, 0.01)), "10 ms"),
             ("HBONGO_PARAMS", 0.0, gate_fn((0.01, 0.01)), "10 ms"),
             ("CLAVES_PARAMS", 150e3, gate_fn((0.01, 0.01)), "10 ms")]
    for name, vr4, gate, label in cases:
        vo, i2, _, _ = run_model(name, n, gate, vr4)
        r, ri = ref.run(getattr(drums, name), FS, n, gate, vr4=vr4)
        e_v, e_i = np.max(np.abs(vo - r)), np.max(np.abs(i2 - ri))
        if verbose:
            print(f"  {name:14s} VR4 {vr4 / 1e3:3.0f}k {label:9s}: V_n3 {vo.min():.3f} ... "
                  f"{vo.max():.3f} V, D2 peak {1e3 * i2.max():.3f} mA; max|model-ref| "
                  f"{e_v:.1e} V, {e_i:.1e} A")
        assert e_v < 1e-8, (name, vr4, label, e_v)
        assert e_i < 1e-12, (name, vr4, label, e_i)


def test_d2_conduction(verbose=False):
    n = FS
    _, i2, _, _ = run_model("BDO_PARAMS", n, gate_fn((0.01, 0.01)), 150e3)
    peak = i2.max()
    late = np.sum(i2[int(0.05 * FS):] > 1e-6)
    _, i2s, _, _ = run_model("BDO_PARAMS", n, gate_fn((0.01, 0.01)), 470e3)
    late_s = np.sum(i2s[int(0.05 * FS):] > 1e-6)
    d = drums.TonalDrum(FS, **drums.BDO_PARAMS)
    d.voice.c7.z -= 1e-3                                  # 1 mV on C7, no trigger
    i2t = np.array([(d.tick(0.0), d.voice.i_d2)[1] for _ in range(FS // 4)])
    if verbose:
        print(f"  BD 150k: D2 peak {1e3 * peak:.3f} mA, samples > 1 uA after 50 ms {late}; "
              f"470k: {late_s} of {n - int(0.05 * FS)}; 1 mV on C7: D2 max {i2t.max():.1e} A")
    assert 0.12e-3 < peak < 0.15e-3, peak
    assert late == 0, late
    assert late_s > 0.1 * (n - int(0.05 * FS)), late_s
    assert i2t.max() < 1e-9, i2t.max()


if __name__ == "__main__":
    test_d2_law()
    test_topology()
    print("drops 0.01/0.1/1 mA: " + ", ".join(f"{drop(i):.4f}" for i in (1e-5, 1e-4, 1e-3)) + " V")
    test_against_reference(verbose=True)
    test_d2_conduction(verbose=True)
    print("OK")
