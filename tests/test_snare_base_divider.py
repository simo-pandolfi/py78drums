#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C1 (and C1b): TR3 base node of the snare.

The base sees C27 through R50 and ground through R51 (and N through C28,
open in DC). The divider must be applied once: with TR3 off the base settles
at V_C27 * R51/(R50+R51) = V_C27/2, and with TR3 on the KCL at the base must
hold with the physical resistors. From C15 the base is a port of the bases
root of pe78/noise_voices.py (RTypeNLRoot, step-limited Newton): it also
converges from a cold start (C1b: the old solver stopped at 1.43 V from 0 V
with V_C27 = 3.85 V).

Usage (from the py78drums root):
    python -m pytest tests/test_snare_base_divider.py
    python tests/test_snare_base_divider.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import NoiseVoices  # noqa: E402

FS = 96000


def _settle(nv, v_c27, n=10000):
    """Bases root with V_C27 held and no noise, until DC (C28/C29 charged)."""
    nv._r50src.set_voltage(v_c27)
    nv._noise_src.set_voltage(0.0)
    for _ in range(n):
        nv._root_b.compute()
    return nv._root_b.v_nodes[0]


def test_open_circuit_base_voltage():
    """TR3 off (Is_b3 = 0): V_base = V_C27 * R51/(R50+R51)."""
    for v_c27 in (0.5, 1.0, 2.0, 3.85):
        nv = NoiseVoices(FS)
        nv.Is_b3 = 0.0
        expected = v_c27 * nv.R51 / (nv.sd.R50 + nv.R51)
        got = _settle(nv, v_c27)
        assert abs(got - expected) <= 1e-6 * expected, (v_c27, got, expected)


def test_base_kcl_and_cold_start():
    """TR3 on, from a cold start: (V_C27 - V)/R50 = V/R51 + Ib(V) in DC."""
    for v_c27 in (1.0, 2.0, 3.85):
        nv = NoiseVoices(FS)                          # all states at 0 V
        v = _settle(nv, v_c27)
        ib = nv.Is_b3 * (np.exp(v / nv.Vt) - 1.0)
        residual = (v_c27 - v) / nv.sd.R50 - v / nv.R51 - ib
        assert abs(residual) < 1e-12, (v_c27, v, residual)
        assert nv._root_b.residuals()[0] < 1e-15
        if v_c27 >= 2.0:
            assert ib > 1e-7, (v_c27, v, ib)          # TR3 conducting


if __name__ == "__main__":
    test_open_circuit_base_voltage()
    test_base_kcl_and_cold_start()
    print("ok")
