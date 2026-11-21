#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for D7 (P4): collector current of TR3 and TR4.

The bases keep their law (Is_b, Vt: base voltages within 3 mV of LTspice,
tools/ltspice_compare.py); the collector current Ic = hFE*Ib must follow
the collector law of the LTspice BC547B card, Ic = IS*exp(Vbe/(NF*VT)),
IS = 2.39e-14 A, NF = 1.008, VT = 25.852 mV (27 C), Early effect left out.
With hFE = 200 (until P4) the collector currents were 1.61 times lower than
LTspice at Vbe = 0.62 V (-4.1 dB on the whole output).

1. Static law: hFE*Is_b*exp(Vbe/Vt) against the card from 0.55 to 0.65 V,
   within 1 %, for TR3 and TR4.
2. In a snare hit and an LC hit (96 kHz): _Ic3 and _Ic4 against the card law
   of the model base voltages, within 1 % whenever Vbe >= 0.55 V.

Usage (from the py78drums root):
    python -m pytest tests/test_collector_gain.py
    python tests/test_collector_gain.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402

FS = 96000
IS, NF, VT = 2.39e-14, 1.008, 25.852e-3          # BC547B card (LTspice)
REL_TOL = 0.01


def card(v):
    return IS * np.expm1(v / (NF * VT))


def test_static_law():
    nv = NoiseVoices(FS)
    v = np.linspace(0.55, 0.65, 101)
    for is_b in (nv.Is_b3, nv.Is_b4):
        ic = nv.hFE * is_b * np.expm1(v / nv.Vt)
        err = np.max(np.abs(ic / card(v) - 1.0))
        print(f"Is_b {is_b:.4g}: max relative error {err:.2e}")
        assert err < REL_TOL, err


def test_hits():
    nv = NoiseVoices(FS)
    n = int(0.2 * FS)
    worst = 0.0
    seen = [0, 0]
    for k in range(n):
        t = (k + 1) / FS
        sd = nv.V_ON if 0.01 <= t < 0.025 else 0.0
        lc = nv.V_ON if 0.1 <= t < 0.115 else 0.0
        nv.tick(sd, lc, 0.0, 0.0, 0.0)
        for j, (vb, ic) in enumerate(((nv._V_b3, nv._Ic3), (nv._V_env, nv._Ic4))):
            if vb >= 0.55:
                seen[j] += 1
                worst = max(worst, abs(ic / card(vb) - 1.0))
    print(f"samples with Vbe >= 0.55 V: TR3 {seen[0]}, TR4 {seen[1]}; max relative error {worst:.2e}")
    assert min(seen) > 1000
    assert worst < REL_TOL, worst


if __name__ == "__main__":
    test_static_law()
    test_hits()
    print("OK")
