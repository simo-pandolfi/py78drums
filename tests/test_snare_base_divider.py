#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C1: TR3 base node of the snare (pe78/snare.py).

The base sees C27 through R50 and ground through R51. The divider must be
applied once: with TR3 off the base sits at V_C27 * R51/(R50+R51) = V_C27/2,
and with TR3 on the KCL at the base must hold with the physical resistors.

Usage (from the py78drums root):
    python -m pytest tests/test_snare_base_divider.py
    python tests/test_snare_base_divider.py
Note: the solver starts from the previous base voltage. From a cold start
(0 V) with V_C27 = 3.85 V it does not converge in 20 iterations (clip of the
exponent at 40); that is a separate, latent issue and is not tested here.

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78.snare import PE78_Snare  # noqa: E402

FS = 48000


def _solve(model, v_c27):
    """Base voltage for a given C27 voltage, through the model's own path."""
    model._V_c27 = v_c27
    v_env = model._V_c27 * model.R51 / (model.R50 + model.R51)   # as _step_envelope
    model._V_base = 0.6   # warm start, as from the previous sample (see note)
    v_base, _, _ = model._solve_base_ebers_moll(v_env)
    return v_base


def test_open_circuit_base_voltage():
    """TR3 off (Is_b = 0): V_base = V_C27 * R51/(R50+R51)."""
    m = PE78_Snare(FS)
    m.Is_b = 0.0
    for v_c27 in (0.5, 1.0, 2.0, 3.85):
        expected = v_c27 * m.R51 / (m.R50 + m.R51)
        got = _solve(m, v_c27)
        assert abs(got - expected) <= 1e-9 * expected, (v_c27, got, expected)


def test_base_kcl():
    """TR3 on: (V_C27 - V)/R50 = V/R51 + Ib(V) at the solved base voltage."""
    m = PE78_Snare(FS)
    for v_c27 in (1.0, 2.0, 3.85):
        v = _solve(m, v_c27)
        ib = m.Is_b * (np.exp(v / m.Vt) - 1.0)
        residual = (v_c27 - v) / m.R50 - v / m.R51 - ib
        assert abs(residual) < 1e-15, (v_c27, v, residual)
        if v_c27 >= 2.0:
            assert ib > 1e-7, (v_c27, v, ib)   # TR3 conducting: the test exercises Ib


if __name__ == "__main__":
    test_open_circuit_base_voltage()
    test_base_kcl()
    print("ok")
