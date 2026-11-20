#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C4 + C4b: twin-T junction, R41, tonal output.

C4b: TwinTDrum._scattering_logic used S = 2 diag(G) A^T M^-1 A - I. For
voltage waves (b = S a) the R-type matrix is S = 2 A^T M^-1 A diag(G) - I.
The old matrix is R^-1 S R: same poles, but the port-8 reading was
V_n3 * R13/R41 instead of V_n3, and port voltages violated KVL.
Since C12 the junction is built by RTypeNLRoot from the MNA of the leaves
(pe78/wdf_nlroot.py), with the CD4011A gate at the root.
C4: R41 = 390 kOhm (Fig. 7) instead of 3.9 MOhm in all tonal voices.
Output: R41 of the N_TONAL voices joined before C21 into RLOAD (VR10).

1. Junction of every tonal voice during a hit (VR4 = 0, 150, 470 kOhm):
   KCL at the four nodes, port voltages = node differences (KVL) and gate
   law, from the leaf waves (RTypeNLRoot.residuals()). The leaf order and
   the node pairs are the ones of the docstring of twint.py.
2. R41 = 390 kOhm in all parameter dictionaries and in leaf 7 (leaf 8 before
   C13, when the trigger source was a twin-T leaf).
3. TonalDrum output stage against the analytic transfer of V_n3 to RLOAD
   (bilinear warping included).

Usage (from the py78drums root):
    python -m pytest tests/test_tonal_output.py
    python tests/test_tonal_output.py

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

from pe78 import drums   # noqa: E402

FS = 48000
R41 = 390e3
VOICES = ("BDO_PARAMS", "CONGA_PARAMS", "HBONGO_PARAMS", "LBONGO_PARAMS", "CLAVES_PARAMS")
NODES = [(2, 0), (0, 3), (0, -1), (2, 1), (1, 3), (1, -1), (2, 3), (3, -1),   # twin-T
         (4, -1), (4, -1), (4, 5), (5, -1), (5, 6)]                           # detector (C13)


def _drum(name, **over):
    return drums.TonalDrum(FS, **dict(getattr(drums, name), **over))


def test_junction_kcl_kvl():
    for name in VOICES:
        for vr4 in (0.0, 150e3, 470e3):
            d = _drum(name, kick_vr4=vr4)
            root = d.voice.root
            leaves = [(leaf, p, q) for leaf, p, q in root.leaves]
            assert [(p, q) for _, p, q in leaves] == NODES, name
            assert [leaf for leaf, _, _ in leaves] == d.voice.components + d.voice.detector, name
            worst = np.zeros(3)
            for k in range(FS // 10):
                d.tick(4.5 if k < 480 else 0.0)
                worst = np.maximum(worst, root.residuals())
            assert worst[0] < 1e-12, f"{name} VR4={vr4:g}: KCL residual {worst[0]:.3g} A"
            assert worst[1] < 1e-9, f"{name} VR4={vr4:g}: KVL residual {worst[1]:.3g} V"
            assert worst[2] < 1e-12, f"{name} VR4={vr4:g}: gate law residual {worst[2]:.3g} A"


def test_r41():
    for name in VOICES:
        assert getattr(drums, name)["kick_load"] == R41, name
        v = _drum(name).voice
        assert v.components[7] is v.r_out
        assert v.r_out.Rp == R41, f"{name}: r_out = {v.r_out.Rp}"


def test_output_stage():
    d = _drum("BDO_PARAMS")
    n, rl, c = d.N_TONAL, d.RLOAD, d.C21
    r_th = R41 / n
    for f in (5.0, 20.0, 96.0, 1000.0):     # integer samples per period at 48 kHz
        d = _drum("BDO_PARAMS")
        n_per = int(round(FS / f))
        n_tot = n_per * int(np.ceil(0.5 * f + 2))            # >= 0.5 s, integer periods
        k = np.arange(n_tot)
        x = np.cos(2 * np.pi * f * k / FS)
        it = iter(x)
        d.voice.process_sample = lambda v: next(it)          # drive V_n3 directly
        y = np.array([d.tick(0.0) for _ in k])
        w = np.exp(-2j * np.pi * f * k / FS)
        win = slice(n_tot - 2 * n_per, n_tot)
        h = np.sum(y[win] * w[win]) / np.sum(x[win] * w[win])
        s = 1j * 2 * FS * np.tan(np.pi * f / FS)
        tau = c * (r_th + rl)
        h_ref = (1.0 / n) * rl / (r_th + rl) * s * tau / (1 + s * tau)
        err = abs(h - h_ref) / abs(h_ref)
        assert err < 1e-6, f"f={f:g} Hz: |H|={abs(h):.6g}, |H_ref|={abs(h_ref):.6g}, err {err:.3g}"


if __name__ == "__main__":
    test_junction_kcl_kvl()
    test_r41()
    test_output_stage()
    print("ok")
