#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""Helpers shared by the tests of pe78/noise_voices.py (not a test module)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
if os.environ.get("PYWDF_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["PYWDF_ROOT"]))

from pe78.noise_voices import NoiseVoices  # noqa: E402
import noise_voices_ref as ref  # noqa: E402

ATTRS = {"c27": "_V_c27", "b3": "_V_b3", "vb5": "_V_b5", "n": "_V_n", "env": "_V_env",
         "lc": "_V_c23", "sc": "_V_c24", "mr": "_V_c26", "a": "_V_a",
         "c3": "_V_col3", "c4": "_V_col4", "o": "_V_out"}


def run_model(gates, t_end, fs, noise=None, nv=None):
    """gates: dict voice -> list of (start, length). Returns t, dict (ref keys), model."""
    m = NoiseVoices(fs) if nv is None else nv
    n = int(round(t_end * fs))
    t = np.arange(1, n + 1) / fs
    out = {k: np.zeros(n) for k in ATTRS}

    def v(name, tt):
        return m.V_ON if any(a <= tt < a + g for a, g in gates.get(name, [])) else 0.0

    for k in range(n):
        m.tick(v("sd", t[k]), v("lc", t[k]), v("sc", t[k]), v("mr", t[k]),
               0.0 if noise is None else noise[k])
        for key, attr in ATTRS.items():
            out[key][k] = getattr(m, attr)
    return t, out, m


def edge_mask(t, gates, win=1e-3):
    keep = np.ones(len(t), dtype=bool)
    for lst in gates.values():
        for a, g in lst:
            for e in (a, a + g):
                keep &= np.abs(t - e) > win
    return keep


def max_err(t, o, tr, r, keys, keep):
    return {k: float(np.abs(o[k] - np.interp(t, tr, r[k]))[keep].max()) for k in keys}
