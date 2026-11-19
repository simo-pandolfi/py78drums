#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
Regression test for C4 + C4b: twin-T scattering matrix, R41, tonal output.

C4b: TwinTDrum._scattering_logic used S = 2 diag(G) A^T M^-1 A - I. For
voltage waves (b = S a) the R-type matrix is S = 2 A^T M^-1 A diag(G) - I.
The old matrix is R^-1 S R: same poles, but the port-8 reading was
V_n3 * R13/R41 instead of V_n3, and port voltages violated KVL.
C4: R41 = 390 kOhm (Fig. 7) instead of 3.9 MOhm in all tonal voices.
Output: R41 of the N_TONAL voices joined before C21 into RLOAD (VR10).

1. S of every tonal voice: port voltages and currents from b = S a satisfy
   KVL (v = A^T v_n) and KCL (A i = inverter current at n3).
2. R41 = 390 kOhm in all parameter dictionaries and in port 8.
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
A = np.array([
    [-1,  1,  1,  0,  0,  0,  0,  0,  0],   # n0 NB_R
    [ 0,  0,  0, -1,  1,  1,  1,  0,  0],   # n1 NC_C
    [ 1,  0,  0,  1,  0,  0,  0,  1,  0],   # n2 NAND_IN
    [ 0, -1,  0,  0, -1,  0,  0, -1,  1],   # n3 NAND_OUT
], dtype=float)


class _Adapter:
    def __init__(self, voice):
        self.voice = voice
        self.S = None

    def get_port_impedances(self):
        return [c.Rp for c in self.voice.components]

    def set_S_matrix(self, S):
        self.S = np.array(S, dtype=float)


def _drum(name, **over):
    return drums.TonalDrum(FS, **dict(getattr(drums, name), **over))


def test_scattering_kvl_kcl():
    rng = np.random.default_rng(0)
    for name in VOICES:
        for vr4 in (0.0, 150e3, 470e3):
            v = _drum(name, kick_vr4=vr4).voice
            ad = _Adapter(v)
            v._scattering_logic(ad)
            R = np.array(ad.get_port_impedances())
            for _ in range(3):
                a = rng.standard_normal(9)
                b = ad.S @ a
                vp = (a + b) / 2.0                 # port voltages
                ip = (b - a) / (2.0 * R)           # currents from the nodes into the ports
                vn = np.linalg.lstsq(A.T, vp, rcond=None)[0]
                scale = np.abs(vp).max()
                kvl = np.abs(A.T @ vn - vp).max() / scale
                i_inv = (v.a_gain * vn[2] - vn[3]) / v.r_o    # inverter current into n3
                kcl = np.abs(A @ ip - np.array([0.0, 0.0, 0.0, i_inv])).max()
                kcl /= np.abs(ip).max()
                assert kvl < 1e-9, f"{name} VR4={vr4:g}: KVL residual {kvl:.3g}"
                assert kcl < 1e-9, f"{name} VR4={vr4:g}: KCL residual {kcl:.3g}"


def test_r41():
    for name in VOICES:
        assert getattr(drums, name)["kick_load"] == R41, name
        v = _drum(name).voice
        assert v.components[8] is v.r_out
        assert v.r_out.Rp == R41, f"{name}: r_out = {v.r_out.Rp}"


def test_output_stage():
    d = _drum("BDO_PARAMS")
    n, rl, c = d.N_TONAL, d.RLOAD, d.C21
    r_th = R41 / n
    for f in (5.0, 20.0, 96.0, 1000.0):     # integer samples per period at 48 kHz
        d = _drum("BDO_PARAMS")
        d.detector.process_sample = lambda v: 0.0            # no trigger
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
    test_scattering_kvl_kcl()
    test_r41()
    test_output_stage()
    print("ok")
