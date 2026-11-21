#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for D11 (P4): M253 trigger outputs of pe78/m253.py.

LTspice (.asc of Fig. 6, B1 = 0, tools/ltspice_compare.py): each output is a
5.2 V pulse through a 1N4148; C23 charges to 4.84-4.86 V and, with the diode
reverse biased, only R47 discharges it: after an LC gate of 17 ms (the gates
of the .asc used for the LTspice traces) V(envelope) stays above 0.5 V for
0.804 s. Until P4 the voices used 4.5 V behind 350 Ohm
and 10 MOhm when off (0.654 s), and the tonal voices took the trigger level
as the source voltage.

1. One definition: NoiseVoices, its snare front end and TwinTDrum use
   pe78/m253.py (4.85 V, 350 Ohm, open when off).
2. The trigger is a gate: a tonal voice gives the same output for a 4.5 V and
   a 10 V trigger.
3. LC hit of 17 ms at 96 kHz: C23 at the end of the gate within 4.80-4.86 V
   and ENV > 0.5 V for 0.80-0.82 s after the gate start.

Usage (from the py78drums root):
    python -m pytest tests/test_m253.py
    python tests/test_m253.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402
from pe78 import drums, m253  # noqa: E402
from pe78.twint import TwinTDrum  # noqa: E402

FS = 96000


def test_one_definition():
    assert (m253.V_ON, m253.R_ON) == (4.85, 350.0) and m253.R_OFF >= 1e12
    nv = NoiseVoices(FS)
    assert (nv.V_ON, nv.R_ON, nv.R_OFF) == (m253.V_ON, m253.R_ON, m253.R_OFF)
    assert (nv.sd.R_ON, nv.sd.R_OFF) == (m253.R_ON, m253.R_OFF)
    assert nv._m253_lc.Rval == m253.R_OFF and nv._m253_mr.Rval == m253.R_OFF
    assert (TwinTDrum.V_ON, TwinTDrum.R_ON, TwinTDrum.R_OFF) == (m253.V_ON, m253.R_ON, m253.R_OFF)


def test_tonal_trigger_is_gate():
    fs = 48000
    y = []
    for level in (4.5, 10.0):
        d = drums.TonalDrum(fs, **drums.BDO_PARAMS)
        y.append(np.array([d.tick(level if k < 480 else 0.0) for k in range(fs // 10)]))
    assert np.max(np.abs(y[0] - y[1])) == 0.0


def test_lc_envelope():
    nv = NoiseVoices(FS)
    t0, g = 0.05, 0.017
    n = int((t0 + 0.9) * FS)
    env = np.zeros(n)
    v_c23_end = None
    for k in range(n):
        t = (k + 1) / FS
        on = t0 <= t < t0 + g
        nv.tick(0.0, nv.V_ON if on else 0.0, 0.0, 0.0, 0.0)
        env[k] = nv._V_env
        if on:
            v_c23_end = nv._V_c23
    t = np.arange(1, n + 1) / FS
    dur = t[np.nonzero(env > 0.5)[0][-1]] - t0
    print(f"C23 at the gate end {v_c23_end:.4f} V; ENV > 0.5 V for {dur:.4f} s (LTspice 0.804 s)")
    assert 4.80 <= v_c23_end <= 4.86, v_c23_end
    assert 0.80 <= dur <= 0.82, dur


if __name__ == "__main__":
    test_one_definition()
    test_tonal_trigger_is_gate()
    test_lc_envelope()
    print("OK")
