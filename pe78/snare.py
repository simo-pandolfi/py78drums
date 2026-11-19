#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 — Snare Drum
Wave Digital Filter implementation
=====================================
Source: Practical Electronics, Jan 1978, Fig. 6 — snare section.

Architecture overview
---------------------
The circuit is partitioned into three sub-circuits, following
the same methodology used in cymb.py (PE78_Cymbals_Maracas).

Sub-circuit A — Trigger front end and envelope  (WDF, R-type root with
                 Newton on the nonlinear ports)
  Schematic (PE 1978 page 24; trigger line confirmed by the user):

    M253 pin 5 ──┬──[C22 0.1u]──┬── TR5 base (BC108B; BC547B card)
                 │              └──[R48 22k]── GND
                 └──[D6 →]── HB line (pin 6) ──[R21 12k]── GND
    TR5 collector: +5 V rail (7805; 4.7 V on load per the schematic)
    TR5 emitter ──[D7 →]──[R49 2.2k]──┬── C27 0.05u ── GND
                                      └──[R50 1M]──> TR3 base (sub-circuit B)

  The M253 output is a switched source (V_ON, R_ON) / (0 V, R_OFF), as in
  cymb.py. C22 and R48 differentiate the gate: TR5 conducts only for the
  first milliseconds after the rising edge, so the C27 peak and the decay do
  not depend on the gate length. After the gate C22 discharges through D6
  and R21 (the input of the HB edge detector, which every snare hit also
  triggers): without that path the next hit would not recharge C27. C12 and
  the rest of the HB edge detector are neglected (C27 peak -0.3%, measured).

  Linear part: leaves M253 source, C22, R48, R21, R49, C27 and the R50 branch
  (a resistive source R50 with V = V_base_TR3 of the previous sample), joined
  by an R-type junction (nodes v1, vb, vd, vc, h). Nonlinear ports at the
  root:
    A  vb -> vd : TR5 base-emitter junction in series with D7, current Ib
    B  vd -> GND: collector current of TR5 entering vd from the rail, -Ic
    D6 v1 -> h
  Unknowns of the Newton iteration: w = V_BE of TR5 and v_D6. With
    Ie = IS*(1 + 1/BF)*(exp(w/(NF*Vt)) - 1),  Ib = Ie/(BF+1),  Ic = Ie - Ib,
    v_A = w + N*Vt*ln(1 + Ie/Is_D)    (D7 carries Ie; always defined)
  the port voltages are matched to the Thevenin of the junction,
  v_N = H*b - Z*i_N (b: reflected waves of the leaves). The emitter node is
  eliminated, so the junction is never singular. Then the node voltages are
  computed and the incident waves a = 2*v - b are sent to the leaves.
  TR5 is forward active only: its base stays below the collector rail.

  The R50 branch decouples the TR3 base by one sample. The coupling goes
  through C27 (port resistance 1/(2*fs*C27) = 208 Ohm at 48 kHz against
  1 MOhm): error on V_C27 <= 3e-4 V against the simultaneous solution.

  V_env = V_C27 * R51 / (R50 + R51)   [V]   (input of sub-circuit B)

Sub-circuit B — TR3 base node  (Ebers-Moll Newton-Raphson)
  The base node is a Thevenin junction of:
    · R50  from node_ENV  (envelope source)
    · R51  to GND         (pull-down / bias)
    · TR3 Vbe junction    (Ebers-Moll nonlinearity)
  C28 noise coupling is handled separately as a bilinear HP IIR.

  V_env is the Thevenin voltage of R50/R51 seen by the base, with
  G_fixed = 1/R50 + 1/R51 = 1/R_th  and  I_fixed = -V_env * G_fixed = -V_C27/R50
  (negative: envelope pushes current INTO the node), the KCL equation is:

      f(V)  = V * G_fixed + I_fixed + Is_b * (exp(V/Vt) - 1) = 0
      f'(V) = G_fixed + Is_b * exp(V/Vt) / Vt  > 0  (monotone)

  Monotonicity guarantees a unique root and quadratic NR convergence.
  Typically 3-5 iterations suffice for convergence to < 1 pV.
  Structurally identical to _solve_env_ebers_moll() in cymb.py.

Sub-circuit C — TR3 collector  (WDF RootRTypeAdaptor)
  Similar to _build_tr4_node() in cymb.py but with an additional port
  for C30 (0.02 uF, collector to GND), which is absent in the cymbal circuit:

      RootRTypeAdaptor([l1, c30, vr9u, tr3_src], _par_scatter)

  where:
    l1      = Inductor(L1_val, fs)          100 mH, ideal (no Rser)
    vr9u    = Resistor(VR9U)                25 kOhm  (snare side of VR9 pot)
    tr3_src = ResistiveVoltageSource(R53)   Norton->Thevenin of the collector
              current gm*v_b in parallel with R53 (4.7 kOhm, L1 || R53 to
              +12V): V_th = gm*v_b*R53, R_th = R53.  R53 is the physical
              resistor, carried by the source; it has no separate port.
              r_o >> R53 is neglected (ideal current source).

  Note on Rser: inductors (Coilcraft PCH-45X-107) have Rser=61.7 Ohm.
  Analysis shows this causes < 0.5 dB difference at the L-C30 resonance
  (3559 Hz) and < 0.1 dB elsewhere in the audio band.  Q ~ 2 (heavily
  damped): timbric impact is imperceptible.  Not modelled, consistent
  with cymb.py.

  Note on VR9U: VR9 (220 kOhm) is a balance pot mixing snare and cymbal
  outputs.  With circuits simulated independently, VR9U (snare side,
  110 kOhm) is the AC load seen by the TR3 collector, exactly as VR9L
  does for TR4 in cymb.py.

Output chain
  V_col -> VR9U -> R55 -> C31 -> out
  C31 bilinear IIR HP, identical in form to cymb.py _make_c31_coeffs().

Noise source  (TR2 / B1, see pe78/noise.py)
  TR2 (BC108B avalanche) noise is shared between snare and cymbal.
  The caller may pass the same noise_sample to both process_sample() calls.
  noise_sample has unit variance; if omitted, an internal N(0,1) sample is
  generated. NOISE_AMP = noise_amp(fs) [A] turns it into the B1 current, which
  reaches the base as a Norton source with resistance VR8U through C28.

Ebers-Moll calibration
  Is_b is calibrated so that V_base_TR3 at steady state with a 4.5 V
  trigger matches LTspice (BC547B Gummel-Poon model).  Derivation:

      Is_b = (V_th - V_target) / (R_th * (exp(V_target/Vt) - 1))

  with R_th = R50 || R51 = 500 kOhm, V_target = LTspice V_base_TR3_ss.
  Starting value: Is_b = 7.035e-17 A (cymb.py value, BC547B IS/BF).
  Recalibrate against LTspice once the reference waveform is available.

Non-linearity separability
  D6, TR5 and D7 are solved together (sub-circuit A). TR3 base (B) takes
  C27 of the current sample and gives C27 its base current through R50 one
  sample later (see A). The collector (C) does not act back on the base.

Known approximations
  TR5: Ebers-Moll forward active (IS, NF, BF); no ISE, RB, RE, capacitances.
  1N4148: Rs and capacitances neglected.
  HB edge detector beyond R21 not modelled.
  Inductor Rser = 61.7 Ohm not modelled.

Component values  (PE 1978 Fig. 6, verified on LTspice ASC)
  Trigger : M253 4.5 V / 350 Ohm on, 10 MOhm off; C22=0.1uF, R48=22kOhm;
            D6 1N4148 to the HB line, R21=12kOhm
  TR5     : BC108B (BC547B card in the .asc), collector to the +5 V rail
            (7805, 4.7 V on load per the schematic; V5 = 5 V in the .asc);
            the rail value does not enter the model (forward active only)
  Envelope: D7 1N4148, R49=2.2kOhm, C27=0.05uF, R50=1MOhm, R51=1MOhm
  TR3     : BC108B, hFE=200, Vt=26mV
  Noise   : C28=4.7nF (node N -> base), B1 = -white(2e6*time)*4uA into N,
            VR8U=90kOhm (N to +12V; VR8 = 100 kOhm trimmer)
  Collector: L1=100mH || R53=4.7kOhm, C30=0.02uF (to GND)
  Output  : VR9U=25kOhm, R55=22kOhm, C31=0.1uF
"""

import numpy as np
import pywdf.core.wdf   as _wdf
from pe78.noise import noise_amp, VR8U as _VR8U
import pywdf.core.rtype as _rtype

Resistor               = _wdf.Resistor
Capacitor              = _wdf.Capacitor
Inductor               = _wdf.Inductor
ResistiveVoltageSource = _wdf.ResistiveVoltageSource
RootRTypeAdaptor       = _rtype.RootRTypeAdaptor


# ---------------------------------------------------------------------------
# Shared parallel scattering  (identical to cymb.py _par_scatter)
# ---------------------------------------------------------------------------
def _par_scatter(adaptor):
    """S[i,j] = 2*G[j]/Gt - delta(i,j).  Sets adaptor.Rp = 1/Gt."""
    G  = [1.0 / p.Rp for p in adaptor.down_ports]
    Gt = sum(G)
    n  = len(G)
    S  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i][j] = 2.0 * G[j] / Gt - (1.0 if i == j else 0.0)
    adaptor.set_S_matrix(S)
    adaptor.Rp = 1.0 / Gt


def _lexp(x, xm=80.0):
    """exp(x) and its derivative, continued linearly above xm (Newton safety)."""
    if x <= xm:
        e = np.exp(x)
        return e, e
    e = np.exp(xm)
    return e * (1.0 + x - xm), e


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------
class PE78_Snare:
    """
    PE78 Snare Drum — Wave Digital Filter.

    TR3 base is resolved each sample via an Ebers-Moll Newton-Raphson
    solver on the base Thevenin node, structurally identical to the TR4
    NR solver in PE78_Cymbals_Maracas (cymb.py).

    Usage
    -----
        snare = PE78_Snare(fs)
        out   = snare.process_sample(v_trigger, noise_sample)

    Observable state (updated after each process_sample call):
        _V_c27  : envelope capacitor voltage [V]
        _V_b5   : TR5 base voltage [V]
        _V_base : TR3 base voltage (NR solution) [V]
        _V_col  : TR3 collector voltage [V]
    """

    # ------------------------------------------------------------------
    # Physical parameters
    # ------------------------------------------------------------------

    # M253 output (pin 5), as in cymb.py
    V_ON      = 4.5         # [V]   output when active
    R_ON      = 350.0       # [Ohm] on-impedance
    R_OFF     = 10e6        # [Ohm] tri-state impedance

    # Trigger front end
    C22       = 0.1e-6      # [F]   gate differentiator
    R48       = 22e3        # [Ohm] TR5 base to GND
    R21       = 12e3        # [Ohm] HB trigger line to GND (reached through D6)
    VT_27C    = 25.852e-3   # [V]   kT/q at 27 C (LTspice), for D6, D7, TR5
    D_IS      = 2.52e-9     # [A]   1N4148 (LTspice standard.dio), D6 and D7
    D_N       = 1.752       # [-]
    TR5_IS    = 2.39e-14    # [A]   BC547B card (LTspice), used for the BC108B
    TR5_NF    = 1.008       # [-]
    TR5_BF    = 294.3       # [-]
    FE_MAXIT  = 100         # Newton iterations per sample (front end)
    FE_TOL    = 1e-10       # [V]   Newton step tolerance

    # Envelope
    R49       = 2_200.0     # [Ohm] D7 -> node_ENV series resistor
    C27       = 0.05e-6     # [F]   hold capacitor
    R50       = 1e6         # [Ohm] node_ENV -> base TR3
    R51       = 1e6         # [Ohm] base TR3 -> GND  (pull-down)

    # TR3 (BC108B)
    Vcc       = 12.0        # [V]   supply
    L1_val    = 100e-3      # [H]   collector inductor  (|| R53 to +12V)
    R53       = 4_700.0     # [Ohm] collector resistor  (|| L1 to +12V)
    C30       = 0.02e-6     # [F]   collector resonance cap (node_C -> GND)
    #                               f_res = 1/(2pi*sqrt(L1*C30)) = 3559 Hz
    #                               Q = (R53||VR9U)/sqrt(L1/C30) ~ 1.8 at
    #                               VR9U = 25 kOhm, ~ 2.0 at 110 kOhm
    C28       = 4.7e-9      # [F]   noise coupling node N -> base
    hFE       = 200         # [-]   DC current gain
    Vt        = 26e-3       # [V]   thermal voltage at 300 K
    # Ebers-Moll base saturation current.
    # Starting value = cymb.py Is_b (BC547B IS/BF ~ 7e-17 A).
    # Recalibrate: Is_b = (V_th - V_tgt) / (R_th * (exp(V_tgt/Vt) - 1))
    # with R_th = R50||R51 = 500 kOhm, V_tgt = LTspice V_base_TR3_ss.
    Is_b      = 7.035e-17   # [A]

    # Internal resistance of the TR3 Thevenin source: the physical R53
    # (mirrors cymb.py R_SRC_TR4)
    R_SRC_TR3 = R53         # [Ohm]

    # Noise (pe78/noise.py); NOISE_AMP [A] is set per instance from fs
    VR8U      = _VR8U       # [Ohm] VR8 between noise node N and +12 V

    # Output
    VR9U_default = 25e3    # [Ohm] snare side of VR9 balance pot (calibrated value)
    R55       = 22e3        # [Ohm] series resistor before C31
    C31       = 0.1e-6      # [F]   AC output coupling cap
    Rload     = 10e3        # [Ohm] pre-amp input impedance

    # r_pi guard rails  (mirrors cymb.py)
    R_PI_MAX  = 10e6        # [Ohm] open-base / cut-off
    R_PI_MIN  = 500.0       # [Ohm] numerical floor (Ic ~ 10 mA); the model is
    #                               linear in the collector (no saturation, no
    #                               high injection): tests/test_linear_region.py

    # ------------------------------------------------------------------
    def __init__(self, fs: int, vr9u: float = None, rload: float = None):
        """
        Parameters
        ----------
        fs    : int    Sample rate [Hz].
        vr9u  : float  VR9U wiper resistance [Ohm]; default 25 kOhm (calibrated value).
        rload : float  Pre-amp input impedance [Ohm]; default 10 kOhm.
        """
        self.fs  = int(fs)
        self._dt = 1.0 / self.fs

        if vr9u  is not None:
            self.VR9U_default = float(vr9u)
        if rload is not None:
            self.Rload = float(rload)

        self.VR9U = float(self.VR9U_default)

        # B1 current per unit of noise_sample [A]
        self.NOISE_AMP = noise_amp(self.fs)

        # Trigger front end and envelope (sub-circuit A)
        self._on     = False
        self._V_c27  = 0.0
        self._V_b5   = 0.0      # TR5 base [V], observable
        self._fe_w   = 0.0      # Newton warm start: V_BE of TR5 [V]
        self._fe_v6  = 0.0      # Newton warm start: V of D6 [V]
        self.fe_iter_max = 0    # largest number of Newton iterations so far
        self._build_frontend()

        # NR warm-start state
        self._V_base = 0.0

        # C28 bilinear HP state  (mirrors cymb.py _v_hp / _v_base_x)
        self._v_base_hp = 0.0
        self._v_base_n  = 0.0   # noise voltage at the TR3 base [V]
        self._v_base_x  = 0.0

        # C31 output HP state
        self._c31_x = 0.0
        self._c31_y = 0.0
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

        # Observable
        self._V_col = 0.0

        # Build WDF collector node
        self._build_tr3_node()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_vr9u(self, value: float) -> None:
        """
        Update VR9U wiper resistance at control rate (not per-sample).
        Mirrors cymb.py set_vr9l(): updates WDF port and C31 coefficients.
        """
        self.VR9U = float(value)
        self._vr9u.set_resistance(self.VR9U)
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

    def process_sample(self,
                       v_trigger: float,
                       noise_sample: float = None) -> float:
        """
        Advance the snare circuit by one sample.

        Parameters
        ----------
        v_trigger    : float
            SNARE_DRUM trigger [V].  Typical: 0.0 V idle, 4.5 V active.
        noise_sample : float or None
            White noise sample (sigma = 1), scaled internally by NOISE_AMP.
            Pass the same value as the cymbal call to share the TR2 bus.
            If None, a fresh N(0,1) sample is drawn internally.

        Returns
        -------
        float
            Audio output [V] after VR9U -> R55 -> C31 high-pass.
        """
        if noise_sample is None:
            noise_sample = np.random.standard_normal()

        # 1. Trigger front end and C27 (WDF, Newton on D6, TR5, D7)
        V_env = self._step_frontend(v_trigger)

        # 2. TR3 base: Ebers-Moll NR
        V_base, gm, r_pi = self._solve_base_ebers_moll(V_env)
        self._V_base = V_base

        # 3. C28 bilinear HP: B1 current -> noise voltage at the base
        v_b = self._step_c28(noise_sample * self.NOISE_AMP, gm)

        # 4. TR3 collector: WDF node
        V_col = self._step_tr3_collector(gm, v_b)
        self._V_col = V_col

        # 5. C31 bilinear IIR HP output coupling
        x = V_col
        y = (self._b_c31[0] * x
             + self._b_c31[1] * self._c31_x
             - self._a_c31[1] * self._c31_y)
        self._c31_x = x
        self._c31_y = y
        return y

    # ------------------------------------------------------------------
    # Private: sub-circuit A — envelope
    # ------------------------------------------------------------------

    # Nodes of the front-end junction and leaf/port incidence: (node+, node-)
    _FE_NODES = ("v1", "vb", "vd", "vc", "h")

    def _build_frontend(self) -> None:
        """
        Leaves (pywdf one-ports) and nonlinear ports of sub-circuit A.

        Leaf k between nodes (p, n) is a Thevenin source: v = b_k + R_k*i_k,
        b_k its reflected wave, i_k the current from p into the leaf.
        """
        self._m253 = ResistiveVoltageSource(self.R_OFF)
        self._c22  = Capacitor(self.C22, self.fs)
        self._r48  = Resistor(self.R48)
        self._r21  = Resistor(self.R21)
        self._r49  = Resistor(self.R49)
        self._c27  = Capacitor(self.C27, self.fs)
        self._r50  = ResistiveVoltageSource(self.R50)   # V = V_base_TR3[n-1]
        V1, VB, VD, VC, H = range(5)
        self._fe_leaves = [
            (self._m253, V1, -1),
            (self._c22,  V1, VB),
            (self._r48,  VB, -1),
            (self._r21,  H,  -1),
            (self._r49,  VD, VC),
            (self._c27,  VC, -1),
            (self._r50,  VC, -1),
        ]
        ports = [(VB, VD), (VD, -1), (V1, H)]          # A, B, D6
        n = len(self._FE_NODES)

        def inc(p, q):
            col = np.zeros(n)
            col[p] = 1.0
            if q >= 0:
                col[q] = -1.0
            return col

        self._fe_AL = np.array([inc(p, q) for _, p, q in self._fe_leaves]).T
        self._fe_AN = np.array([inc(p, q) for p, q in ports]).T
        self._fe_cache = {}

    def _frontend_matrices(self):
        """Junction matrices for the current leaf resistances (cached)."""
        R = np.array([leaf.Rp for leaf, _, _ in self._fe_leaves])
        key = tuple(R)
        m = self._fe_cache.get(key)
        if m is None:
            AL, AN = self._fe_AL, self._fe_AN
            Ginv = np.linalg.inv(AL @ np.diag(1.0 / R) @ AL.T)
            B = AL @ np.diag(1.0 / R)          # source term: B @ b
            m = (Ginv @ B, Ginv @ AN, AN.T @ Ginv @ B, AN.T @ Ginv @ AN)
            self._fe_cache[key] = m
        return m

    def _step_frontend(self, v_trigger: float) -> float:
        """
        Advance sub-circuit A by one sample. Returns V_env = V_C27*R51/(R50+R51).

        Newton on (w, v6): w = V_BE of TR5, v6 = voltage of D6. Port currents
        i = (Ib, -Ic, I_D6); port voltages from the junction v_N = H*b - Z*i.
        """
        on = v_trigger > 0.5
        if on != self._on:
            self._m253.set_resistance(self.R_ON if on else self.R_OFF)
            self._on = on
        self._m253.set_voltage(self.V_ON if on else 0.0)
        self._r50.set_voltage(self._V_base)

        b = np.array([leaf.propagate_reflected_wave() for leaf, _, _ in self._fe_leaves])
        Kb, Ki, H, Z = self._frontend_matrices()
        h = H @ b

        vt_t = self.TR5_NF * self.VT_27C
        vt_d = self.D_N * self.VT_27C
        k_e = self.TR5_IS * (1.0 + 1.0 / self.TR5_BF)
        k_b = 1.0 / (self.TR5_BF + 1.0)
        w, v6 = self._fe_w, self._fe_v6
        for it in range(1, self.FE_MAXIT + 1):
            ex, dex = _lexp(w / vt_t)
            ie, die = k_e * (ex - 1.0), k_e * dex / vt_t
            ib, dib = k_b * ie, k_b * die
            ic, dic = ie - ib, die - dib
            u = 1.0 + ie / self.D_IS
            vA = w + vt_d * np.log(u)
            dvA = 1.0 + vt_d * die / (self.D_IS * u)
            e6, de6 = _lexp(v6 / vt_d)
            i6, di6 = self.D_IS * (e6 - 1.0), self.D_IS * de6 / vt_d
            iN = (ib, -ic, i6)
            fA = vA - (h[0] - Z[0, 0] * iN[0] - Z[0, 1] * iN[1] - Z[0, 2] * iN[2])
            f6 = v6 - (h[2] - Z[2, 0] * iN[0] - Z[2, 1] * iN[1] - Z[2, 2] * iN[2])
            jAw = dvA + Z[0, 0] * dib - Z[0, 1] * dic
            jA6 = Z[0, 2] * di6
            j6w = Z[2, 0] * dib - Z[2, 1] * dic
            j66 = 1.0 + Z[2, 2] * di6
            det = jAw * j66 - jA6 * j6w
            dw = -(fA * j66 - jA6 * f6) / det
            d6 = -(jAw * f6 - j6w * fA) / det
            dw = min(max(dw, -0.2), 0.2)            # step limit [V]
            d6 = min(max(d6, -0.2), 0.2)
            w += dw
            v6 += d6
            if abs(dw) < self.FE_TOL and abs(d6) < self.FE_TOL:
                break
        else:
            raise RuntimeError("snare front end: Newton did not converge")
        if it > self.fe_iter_max:
            self.fe_iter_max = it
        self._fe_w, self._fe_v6 = w, v6

        # port currents at the solution, node voltages, incident waves
        ex, _ = _lexp(w / vt_t)
        ie = k_e * (ex - 1.0)
        ib = k_b * ie
        e6, _ = _lexp(v6 / vt_d)
        iN = np.array([ib, -(ie - ib), self.D_IS * (e6 - 1.0)])
        v = Kb @ b - Ki @ iN
        vL = self._fe_AL.T @ v
        for (leaf, _, _), vk, bk in zip(self._fe_leaves, vL, b):
            leaf.accept_incident_wave(2.0 * vk - bk)

        self._V_b5  = v[1]
        self._V_c27 = self._c27.wave_to_voltage()
        return self._V_c27 * (self.R51 / (self.R50 + self.R51))

    # ------------------------------------------------------------------
    # Private: sub-circuit B — TR3 base Ebers-Moll NR
    # ------------------------------------------------------------------

    def _solve_base_ebers_moll(self, V_env: float):
        """
        Resolve TR3 base voltage and operating point via Ebers-Moll NR.

        Base node KCL (currents leaving node positive):
            f(V)  = V * G_fixed + I_fixed + Is_b * (exp(V/Vt) - 1) = 0
            f'(V) = G_fixed + Is_b * exp(V/Vt) / Vt  > 0

        G_fixed = 1/R50 + 1/R51
        I_fixed = -V_env * G_fixed = -V_C27 / R50
                                 (V_env is the Thevenin voltage of R50/R51;
                                  the envelope injects current into the node)

        Structurally identical to cymb.py _solve_env_ebers_moll(), with
        the two-resistor Thevenin (R50, R51) replacing the multi-branch
        ENV junction.

        Returns  (V_base, gm, r_pi).
        """
        G_fixed = 1.0 / self.R50 + 1.0 / self.R51
        I_fixed = -V_env * G_fixed      # Norton of the R50/R51 Thevenin (= -V_C27/R50)

        Vt   = self.Vt
        Is_b = self.Is_b

        # NR with warm start from previous solution
        V = self._V_base
        for _ in range(20):
            ex = np.exp(np.clip(V / Vt, -40.0, 40.0))
            f  = V * G_fixed + I_fixed + Is_b * (ex - 1.0)
            fp = G_fixed + Is_b * ex / Vt
            dV = -f / fp
            V += dV
            if abs(dV) < 1e-12:
                break

        # Collector current and small-signal parameters
        ex_sol = np.exp(np.clip(V / Vt, -40.0, 40.0))
        Ic     = self.hFE * Is_b * ex_sol
        Ic     = max(Ic, 1e-15)                         # numerical floor

        gm    = Ic / Vt
        r_pi  = float(np.clip(self.hFE * Vt / Ic,
                               self.R_PI_MIN, self.R_PI_MAX))
        return V, gm, r_pi

    # ------------------------------------------------------------------
    # Private: C28 bilinear high-pass  (noise -> base coupling)
    # ------------------------------------------------------------------

    def _step_c28(self, noise_i: float, gm: float) -> float:
        """
        Advance the C28 coupling by one sample.

        noise_i is the B1 current [A] into node N. Seen from C28, the
        Norton source (noise_i, VR8U) is a Thevenin source
        (noise_i*VR8U, VR8U); with r_base = r_pi || R51 at the base:

          v_b = noise_i*VR8U * r_base/(VR8U + r_base) * H(s)
          H(s) = s*tau / (1 + s*tau),  tau = C28 * (VR8U + r_base)

        Bilinear transform of H (k = 2*fs*tau):
          alpha = k / (1 + k)
          y[n]  = alpha*(x[n] - x[n-1]) + (2*alpha - 1)*y[n-1]

        The load of C29 (cymbal side) on node N is not modelled.
        Returns the noise voltage at the TR3 base [V].
        """
        if gm > 1e-12:
            r_pi   = self.hFE / gm
            r_base = r_pi * self.R51 / (r_pi + self.R51)
        else:
            r_base = self.R51

        r_ser   = self.VR8U + r_base
        tau_c28 = self.C28 * r_ser
        k       = 2.0 * self.fs * tau_c28
        alpha   = k / (1.0 + k)

        v_th            = noise_i * self.VR8U
        x_new           = v_th - self._v_base_x
        self._v_base_hp = alpha * x_new + (2.0 * alpha - 1.0) * self._v_base_hp
        self._v_base_x  = v_th
        self._v_base_n  = self._v_base_hp * r_base / r_ser
        return self._v_base_n

    # ------------------------------------------------------------------
    # Private: sub-circuit C — TR3 collector WDF
    # ------------------------------------------------------------------

    def _build_tr3_node(self) -> None:
        """
        Build TR3 collector WDF tree.

        Unlike cymb.py _build_tr4_node(), the snare collector has C30
        (0.02 uF) connected from node_C to GND — absent in the cymbal
        circuit.  C30 creates a resonance with L1 at f_res = 3559 Hz
        (Q ~ 1.8 with VR9U = 25 kOhm) that shapes the characteristic "crack" of the snare.

        Structure:
            RootRTypeAdaptor([l1, c30, vr9u, tr3_src], _par_scatter)

        All ports shunt to AC-GND (+12V rail bypassed for small signal).
        R53 is the internal resistance of tr3_src, not a separate port.
        V_col is read from the VR9U port (parallel junction: every port has
        the node voltage).  A passive port is used on purpose: with the
        original pywdf (root R-type compute order, E1) the source port
        voltage is wrong, while passive ports are correct.
        """
        self._l1      = Inductor(self.L1_val, self.fs)
        self._c30     = Capacitor(self.C30, self.fs)
        self._vr9u    = Resistor(self.VR9U)
        self._tr3_src = ResistiveVoltageSource(self.R_SRC_TR3)
        self._tr3_src.set_voltage(0.0)

        self._tr3_node = RootRTypeAdaptor(
            [self._l1, self._c30, self._vr9u, self._tr3_src],
            _par_scatter)

    def _step_tr3_collector(self, gm: float, v_b: float) -> float:
        """
        Advance TR3 collector WDF node by one sample.

        Norton -> Thevenin conversion:
          V_th = gm * v_b * R_SRC_TR3
          R_th = R_SRC_TR3  (= R53, the physical collector resistor)

        Exact for the current source gm*v_b in parallel with R53; only
        r_o (> 100 kOhm, >> R53) is neglected.  Same convention as
        cymb.py R_SRC_TR4.

        Returns TR3 collector voltage V_col [V].
        """
        self._tr3_src.set_voltage(gm * v_b * self.R_SRC_TR3)
        self._tr3_node.compute()
        return self._vr9u.wave_to_voltage()   # node voltage, see _build_tr3_node

    # ------------------------------------------------------------------
    # Private: C31 bilinear IIR HP coefficients
    # ------------------------------------------------------------------

    def _make_c31_coeffs(self):
        """
        Bilinear IIR HP for C31 output coupling.

        Identical to cymb.py _make_c31_coeffs(), with VR9U in place of VR9L:
          tau_p = C31 * (VR9U + R55 + Rload)
          tau_z = C31 * Rload

        Bilinear transform  (k = 2*fs):
          b = [k*tau_z / (1 + k*tau_p),  -k*tau_z / (1 + k*tau_p)]
          a = [1,                         (1 - k*tau_p) / (1 + k*tau_p)]
        """
        Rs    = self.VR9U + self.R55
        Rl    = self.Rload
        C     = self.C31
        tau_p = C * (Rs + Rl)
        tau_z = C * Rl
        k     = 2.0 * self.fs
        b0    =  k * tau_z
        b1    = -k * tau_z
        a0    = 1.0 + k * tau_p
        a1    = 1.0 - k * tau_p
        return np.array([b0 / a0, b1 / a0]), np.array([1.0, a1 / a0])


# ---------------------------------------------------------------------------
# SnareDrum wrapper  —  public interface for wdf_snare_test.py
# ---------------------------------------------------------------------------

class SnareDrum:
    """
    Thin public wrapper around PE78_Snare.

    Exposes the tick() interface expected by wdf_snare_test.py and the
    parent drum machine.

    Shared-noise usage (TR2 bus with cymbal):
    ::
        noise = np.random.standard_normal()
        cymbal_out = cymbal.process_sample(v_lc, v_sc, v_mr, noise)
        snare_out  = snare.tick(v_trigger, noise_sample=noise)
    """

    def __init__(self, fs: int, vr9u: float = None, rload: float = None):
        self.model = PE78_Snare(fs, vr9u=vr9u, rload=rload)

    def set_vr9u(self, value: float) -> None:
        self.model.set_vr9u(value)

    def tick(self,
             v_trigger: float,
             noise_sample: float = None) -> float:
        """
        Per-sample entry point.

        Parameters
        ----------
        v_trigger    : float   Gate voltage [V]: 4.5 V on, 0 V off.
        noise_sample : float   sigma~1 noise (shared TR2 sample); auto if None.

        Returns
        -------
        float   Audio output [V].
        """
        return self.model.process_sample(v_trigger, noise_sample)