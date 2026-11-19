#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Check of the ENV <-> maracas coupling (C9; from C15 in pe78/noise_voices.py).

The bases root (ENV) and the maracas root see each other through Thevenin
sources of the previous sample (one-sample delay both ways, R46 in both
roots, static R_th). C9 kept this structure (decision D1-A) after measuring
its error against a reference that solves the whole section at once. This
test bounds that error at 96 kHz and 48 kHz against tools/noise_voices_ref.py
so that later changes cannot make it grow unnoticed.

Measured at C15 (edges +-1 ms excluded): maracas alone V_ENV 0.86 / 1.44 mV,
V_C26 0.93 / 2.70 mV (96 / 48 kHz); LC, SC and MR overlapping V_C26 0.78 /
2.31 mV.

Usage (from the py78drums root):
    python -m pytest tests/test_cymb_coupling.py
    python tests/test_cymb_coupling.py

pywdf is taken from the environment; set PYWDF_ROOT to the folder that
contains the pywdf package to use a specific copy (for example pywdf-ref).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _nv_common import ref, run_model, edge_mask, max_err  # noqa: E402

CASES = {
    "mr_alone": ({"mr": [(0.05, 0.015)]}, 0.4),
    "overlap": ({"lc": [(0.05, 0.1)], "sc": [(0.08, 0.1)], "mr": [(0.11, 0.1)]}, 0.6),
}
TOL = {96000: 2e-3, 48000: 4e-3}        # [V] on V_ENV, V_C26, V_C23


def _case(name):
    g, t_end = CASES[name]
    tr, r = ref.simulate(g, t_end)
    out = {}
    for fs, tol in TOL.items():
        t, o, _ = run_model(g, t_end, fs)
        err = max_err(t, o, tr, r, ("env", "mr", "lc"), edge_mask(t, g))
        for k, d in err.items():
            assert d < tol, (name, fs, k, d)
        out[fs] = err
    return out


def test_mr_alone():
    _case("mr_alone")


def test_overlap():
    _case("overlap")


if __name__ == "__main__":
    for name in CASES:
        for fs, err in _case(name).items():
            print(f"{name:9s} {fs:6d} Hz: " + ", ".join(f"{k} {v*1e3:.2f} mV" for k, v in err.items()))
    print("ok")
