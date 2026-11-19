#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 — Long Cymbal + Short Cymbal + Maracas
MNA/Werner Wave Digital Filter implementation
==============================================
Source: Practical Electronics, Jan 1978, Fig. 6 — bottom section.

Architecture overview
---------------------
Sub-circuit A — Envelope node (ENV):
  _ParallelNLRoot([ser_lc, ser_sc, rvsrc_mr_port, r52]): parallel junction
  with the TR4 base (Shockley) as the nonlinear element at the root, solved
  by Newton-Raphson on the Thevenin of the four subtrees (see below).

  LC branch:
    ser_lc = Series(Parallel(rvsrc_lc, c23), r47)
    Topology: PIN21──[R_src]──┬──[R47, 1MΩ]──> ENV
                              └─[C23, 0.33µF]── GND

  SC branch (rvsrc_sc voltage inverted: source on GND side):
    ser_sc = Series(r44, Parallel(c24, Series(r42, rvsrc_sc)))
    Topology: PIN20──[R_src]──[R42, 10kΩ]──┬──[R44, 1MΩ]──> ENV
                                            └─[C24, 68nF]── GND

  MR → ENV port (one-sample delayed Thevenin of the MR branch):
    rvsrc_mr_port: R_th = R46, V_th = −V_C26[n-1]
    R_th = R46 only (not R45+R46): C26 sits at n_MR and separates
    R45 (D9 side) from R46 (ENV side) in the Thevenin seen from ENV.

  Coupling ENV <-> maracas branch (sub-circuit M): one-sample delay both
  ways, R46 in both roots, resistive (static) R_th of ENV. Measured against
  tools/cymb_env_ref.py (all simultaneous), edges +-1 ms excluded (C9):
  V_ENV <= 1.1 mV and V_C26 <= 0.9 mV at 96 kHz, <= 1.9 / 2.7 mV at 48 kHz;
  a dynamic R_th (C23, C24 included) changes the results by < 0.01 mV.
  Checked by tests/test_cymb_coupling.py.

  r52           : ENV pull-down to GND (1 MΩ)

Sub-circuit B — TR4 collector:
  RootRTypeAdaptor([l2, vr9l, tr4_src])
  R54 is the internal resistance of tr4_src (Norton->Thevenin of the
  collector current gm*v_b in parallel with R54), not a separate port.
  Driven by V_env from sub-circuit A (one-sample delay).

Sub-circuit M — Maracas branch (C25 in series, D8, D9)
------------------------------------------------------
Schematic (PE 1978 Fig. 6, .asc):

    PIN19 ─[M253]─ S ──[C25 0.22µF]── A ──[D9 →]── K ──[R45 100k]── n_MR
                   │                  │                              │
               [R43 22k]           [D8] anode at GND            [C26 0.1µF]
                   │                  │                              │
                  GND                GND                    [R46 470k]─> ENV

The rising edge passes through C25 and charges C26 through D9 and R45; the
charge is shared between C25 and C26 (no DC path to C26), so V_C26 peaks
and then decays even while the gate is high. At the end of the gate A goes
negative and D8 clamps it (about -0.4 V).

Implementation: pe78.wdf_nlroot.RTypeNLRoot. Leaves (pywdf): M253 source
(V_ON/R_ON, 0 V/R_OFF), R43, C25, R45, C26 and the ENV branch, a resistive
source R46 + R_th with the open-circuit voltage of ENV (previous sample):
    R_th = R52 ‖ (R47 + R_src_lc) ‖ (R42 + R44 + R_src_sc) ‖ r_pi
    V_th = V_env − R_th · (V_C26 − V_env)/R46
Nonlinear ports: D8 (A–GND, anode at GND) and D9 (A–K), 1N4148 as LTspice
standard.dio (Is 2.52 nA, N 1.752), solved together by Newton-Raphson (they
are coupled through A: C25 is about 24 Ohm at 96 kHz).

TR4 base loading — nonlinear root, Newton-Raphson (Werner 2015 §IV)
--------------------------------------------------------------------
At each sample the root collects the reflected waves a_k of the four
subtrees (a_vals). Seen from the TR4 base they are a Thevenin; by Millman's
theorem (pywdf sign: the junction voltage is −V_env, see the series
adaptors of the LC and SC branches):

    V_env = −I_fixed / G_fixed     without base current
    where  G_fixed = Σ_k G_k,   I_fixed = Σ_k G_k · a_vals_k

The BC547B base current follows the Shockley diode equation:
    Ib(V) = Is_b · (exp(V/Vt) − 1)

Substituting G_total = G_fixed + Ib(V)/V into the Millman equation yields:

    f(V) = V · G_fixed + I_fixed + Is_b · (exp(V/Vt) − 1) = 0

This is solved per-sample with Newton-Raphson:
    f'(V) = G_fixed + Is_b · exp(V/Vt) / Vt  > 0 for all V  (monotone)

Monotonicity of f guarantees:
  - A unique solution from any starting point (no bistability)
  - Quadratic NR convergence (typically 3-5 iterations)
  - No hard Vbe threshold → smooth gm(t) → no amplitude artifact at cutoff

Is_b is calibrated so that the DC operating point with LC trigger matches
LTspice:  V_env_ss ≈ 620 mV.  This gives:
  Is_b = (V_th − V_target) / (R_th · (exp(V_target/Vt) − 1))
       ≈ 7.035 × 10⁻¹⁷ A   (close to IS/BF of the BC547B SPICE model)

After solving, the root sends to every subtree the incident wave of the
junction voltage, b_k = 2·(−V_env) − a_k: C23, C24 and the MR port see the
node voltage with the base current drawn (KCL exact, no delay). Before C7
the base was a linear port r_pi of the previous sample inside the scatter,
so the subtrees saw a node at about 68 mV while the NR gave 620 mV.
r_pi = hFE·Vt/Ic (incremental) is kept for the D9-branch ENV Thevenin of
the next sample and for the noise path.

Quantitative agreement with LTspice reference
----------------------------------------------
  V_ENV during LC trigger ≈ 620 mV  (LTspice: ~620 mV, error < 1 %)
  No period-2 oscillation at any trigger voltage.
  SC output decay smoother: no hard cutoff at Vbe threshold.
"""

import numpy as np
import pywdf.core.wdf as _wdf
import pywdf.core.rtype as _rtype
from pe78.noise import noise_amp, VR8U as _VR8U
from pe78.wdf_nlroot import RTypeNLRoot, diode_law

Resistor               = _wdf.Resistor
Capacitor              = _wdf.Capacitor
Inductor               = _wdf.Inductor
ResistiveVoltageSource = _wdf.ResistiveVoltageSource
SeriesAdaptor          = _wdf.SeriesAdaptor
ParallelAdaptor        = _wdf.ParallelAdaptor
RootRTypeAdaptor       = _rtype.RootRTypeAdaptor


# ---------------------------------------------------------------------------
# Shared parallel scattering (star junction)
# ---------------------------------------------------------------------------
def _par_scatter(adaptor):
    """S[i][j] = 2·G[j]/Gt − δ(i,j).  Sets adaptor.Rp = 1/Gt."""
    G  = [1.0 / p.Rp for p in adaptor.down_ports]
    Gt = sum(G)
    n  = len(G)
    S  = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i][j] = 2.0 * G[j] / Gt - (1.0 if i == j else 0.0)
    adaptor.set_S_matrix(S)
    adaptor.Rp = 1.0 / Gt


class _ParallelNLRoot(RootRTypeAdaptor):
    """
    Parallel junction with a nonlinear one-port at the root.

    compute(): collects the reflected waves a_k of the subtrees, calls
    solve(G_fixed, I_fixed) -> v (junction voltage, pywdf orientation) with
        G_fixed = Σ G_k,  I_fixed = Σ G_k·a_k,  G_k = 1/Rp_k,
    and sends b_k = 2·v − a_k to every subtree. The current into the
    nonlinear element is Σ G_k·(a_k − v); solve() must satisfy its law.
    """

    def __init__(self, down_ports: list, solve) -> None:
        self._solve = solve
        super().__init__(down_ports, lambda adaptor: None)

    def compute(self) -> None:
        n = len(self.down_ports)
        G_fixed = 0.0
        I_fixed = 0.0
        for i in range(n):
            a = self.down_ports[i].propagate_reflected_wave()
            self.a_vals[i] = a
            g = 1.0 / self.down_ports[i].Rp
            G_fixed += g
            I_fixed += g * a
        v = self._solve(G_fixed, I_fixed)
        for i in range(n):
            self.b_vals[i] = 2.0 * v - self.a_vals[i]
            self.down_ports[i].accept_incident_wave(self.b_vals[i])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class PE78_Cymbals_Maracas:
    """
    PE78 Long Cymbal / Short Cymbal / Maracas — Wave Digital Filter.

    -----------------------------
    TR4 base loading is resolved each sample via an Ebers-Moll Newton-Raphson
    solver derived from Millman's theorem on the parallel WDF junction.
    The Shockley equation Ib = Is_b·(exp(V/Vt)−1) is monotone → unique root,
    no bistability, no hard Vbe threshold, smooth gm decay.

    V_ENV converges to ~620 mV during LC trigger (LTspice reference: ~620 mV).

    Usage
    -----
        drum = PE78_Cymbals_Maracas(fs)
        out  = drum.process_sample(v_lc, v_sc, v_mr, noise)

    Observable state (updated after each process_sample call):
        _V_env  : ENV node voltage [V]
        _V_c23  : C23 voltage [V]
        _V_c24  : C24 voltage [V]
        _V_c26  : C26 voltage [V]
        _V_col  : TR4 collector voltage [V]
    """

    # -----------------------------------------------------------------------
    # Physical parameters  (PE 1978 Fig. 6, bottom section)
    # -----------------------------------------------------------------------
    V_ON  = 4.5        # M253 output when conducting [V]
    R_ON  = 350.0      # M253 on-impedance [Ω]
    R_OFF = 10e6       # M253 tri-state impedance [Ω]

    R47   = 1e6        # LC: C23 discharge resistor; tau = R47·C23 = 330 ms
    C23   = 0.33e-6

    R42   = 10e3       # SC: series current limiter
    C24   = 68e-9      # SC: hold cap; tau = R44·C24 = 68 ms
    #C24   = 270e-9    # to increase tau time to about 170 ms

    R44   = 1e6

    R43   = 22e3       # MR: D9 anode pull-down
    C25   = 0.22e-6    # MR: series coupling cap (WDF leaf)
    R45   = 100e3      # MR: D9 cathode series
    C26   = 0.1e-6     # MR: hold cap; tau = R46·C26 = 47 ms
    R46   = 470e3

    R52   = 1e6        # ENV pull-down / TR4 base bias

    # TR4 (BC547B)
    Vcc       = 12.0
    R54       = 4700.0
    L2_val    = 100e-3
    C29       = 4.7e-9     # noise coupling node N -> TR4 base
    hFE       = 200
    Vt        = 26e-3
    # Ebers-Moll base saturation current, calibrated so that
    # V_ENV_ss ≈ 620 mV with LC trigger (matches LTspice BC547B operating point).
    # Derivation: Is_b = (V_th − V_target) / (R_th · (exp(V_target/Vt) − 1))
    # with V_th ≈ 1200 mV, R_th ≈ 363 kΩ, V_target = 620 mV.
    Is_b      = 7.035e-17  # [A]  ≈ IS/BF of BC547B SPICE model
    # Vbe_th is kept for gm reconstruction in the test script and for the
    # D9-branch ENV Thevenin (used only as a fallback guard rail).
    Vbe_th    = 0.58       # [V]  approximate; not used in the NR solver
    R_SRC_TR4 = R54
    # Noise (pe78/noise.py); NOISE_AMP [A] is set per instance from fs
    VR8U      = _VR8U      # [Ω] VR8 between noise node N and +12 V

    # 1N4148 (LTspice standard.dio), D8 and D9
    VT_27C = 25.852e-3 # [V]  kT/q at 27 C (LTspice)
    D_IS   = 2.52e-9   # [A]
    D_N    = 1.752     # [-]

    VR9L_default = 195e3
    R55   = 22e3
    C31   = 0.1e-6
    Rload = 10e3

    # r_pi guard rails
    R_PI_MAX = 10e6    # open-base / cut-off
    R_PI_MIN = 500.0   # numerical floor (Ic ~ 10 mA); the model is linear in
    #                    the collector (no saturation, no high injection):
    #                    tests/test_linear_region.py

    # -----------------------------------------------------------------------
    def __init__(self, fs: int, vr9l: float = None):
        self.fs  = int(fs)
        self._dt = 1.0 / fs
        self.VR9L = float(vr9l) if vr9l is not None else self.VR9L_default

        # B1 current per unit of noise_sample [A]
        self.NOISE_AMP = noise_amp(self.fs)

        self._on_lc = self._on_sc = self._on_mr = False

        # Maracas branch observables (C25 voltage S-A, node A)
        self._V_c25  = 0.0
        self._V_a    = 0.0

        # One-sample delays
        self._V_env  = 0.0    # V_ENV[n-1]  (used to drive TR4 collector)
        self._V_c26  = 0.0    # C26[n-1]    (MR → ENV Thevenin voltage)

        # r_pi from the last NR solve — feeds the D9-branch ENV Thevenin
        self._r_pi_prev = 10e6      # starts at open-base (cut-off)

        # C29 HP state and TR4 base noise voltage
        self._v_hp     = 0.0
        self._v_base_x = 0.0
        self._v_base   = 0.0
        
        # C31 output HP
        self._c31_x = 0.0
        self._c31_y = 0.0
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

        # Observable
        self._V_c23  = 0.0
        self._V_c24  = 0.0
        self._V_col  = 0.0

        self._build_env_node()
        self._build_mr_tree()
        self._build_tr4_node()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def set_vr9l(self, value: float):
        """Update VR9L at control rate (between triggers, not per-sample)."""
        self.VR9L = float(value)
        self._vr9l.set_resistance(self.VR9L)
        self._b_c31, self._a_c31 = self._make_c31_coeffs()

    def process_sample(self, v_trig_lc: float, v_trig_sc: float,
                       v_trig_mr: float, noise_sample: float = None) -> float:
        """
        Advance the circuit by one sample.

        Parameters
        ----------
        v_trig_lc, v_trig_sc, v_trig_mr : float
            M253 trigger voltages: 0.0 V = idle, 4.5 V = active.
        noise_sample : float or None
            White noise sample (σ = 1), scaled internally by NOISE_AMP.
            If None, an internal N(0,1) sample is generated (as in snare.py).

        Returns
        -------
        float
            Audio output [V] after C31 high-pass coupling.
        """
        if noise_sample is None:
            noise_sample = np.random.standard_normal()

        on_lc = v_trig_lc > 0.5
        on_sc = v_trig_sc > 0.5
        on_mr = v_trig_mr > 0.5

        # ---- 1. Source impedance transitions (only on edges) ------------
        if on_lc != self._on_lc:
            self._rvsrc_lc.set_resistance(self.R_ON if on_lc else self.R_OFF)
            self._on_lc = on_lc
        if on_sc != self._on_sc:
            self._rvsrc_sc.set_resistance(self.R_ON if on_sc else self.R_OFF)
            self._on_sc = on_sc
        if on_mr != self._on_mr:
            self._m253_mr.set_resistance(self.R_ON if on_mr else self.R_OFF)
            self._on_mr = on_mr

        # ---- 2. Source voltages ----------------------------------------
        self._rvsrc_lc.set_voltage(self.V_ON if on_lc else 0.0)
        self._rvsrc_sc.set_voltage(-(self.V_ON if on_sc else 0.0))
        self._m253_mr.set_voltage(self.V_ON if on_mr else 0.0)

        # ---- 4. MR → ENV Thevenin (one-sample delayed C26) -------------
        # R_th = R46: ENV sees n_MR through R46 only.  R45 is on the D9 side
        # of n_MR and is separated from ENV by C26, so it does not appear in
        # the Thevenin.  V_th = −V_C26 (sign: R52 pulls ENV to GND).
        self._rvsrc_mr_port.set_resistance(self.R46)
        self._rvsrc_mr_port.set_voltage(-self._V_c26)

        # ---- 5. ENV seen from n_MR (Thevenin, previous sample) ----------
        #  Includes r_pi (incremental) of the PREVIOUS sample.
        R_src_lc = self._rvsrc_lc.Rval
        R_src_sc = self._rvsrc_sc.Rval
        R_env_th = 1.0 / (
            1.0 / (self.R47 + R_src_lc)
            + 1.0 / (self.R42 + self.R44 + R_src_sc)
            + 1.0 / self.R52
            + 1.0 / self._r_pi_prev          # TR4 base loading
        )
        # Open-circuit voltage of ENV without the MR current: V_env[n-1] already
        # contains the drop of the R46 current on R_env_th (C7, D4-B).
        i_r46 = (self._V_c26 - self._V_env) / self.R46     # n_MR -> ENV [A]
        self._env_mr.set_resistance(self.R46 + R_env_th)
        self._env_mr.set_voltage(self._V_env - R_env_th * i_r46)

        # ---- 6. Maracas branch: R-type root, Newton on D8 and D9 ---------
        self._root_mr.compute()
        self._V_c26 = self._c26.wave_to_voltage()
        self._V_c25 = self._c25.wave_to_voltage()
        self._V_a   = self._root_mr.v_nodes[1]

        # ---- 7-8. ENV node: nonlinear root (Ebers-Moll NR) --------------
        #  Collect the subtree waves, solve V_env, send the waves back.
        self._root_env.compute()
        V_env_new, gm_new, r_pi_new = self._env_solution

        self._V_env  = V_env_new
        self._V_c23  = self._c23.wave_to_voltage()
        self._V_c24  = self._c24.wave_to_voltage()
        self._r_pi_prev = r_pi_new

        # ---- 9. TR4 collector (sub-circuit B, driven by V_env_new) -----
        V_col       = self._step_tr4(V_env_new, gm_new, noise_sample * self.NOISE_AMP)
        self._V_col = V_col

        # ---- 10. C31 IIR high-pass output coupling ----------------------
        x = V_col
        y = (self._b_c31[0] * x
             + self._b_c31[1] * self._c31_x
             - self._a_c31[1] * self._c31_y)
        self._c31_x = x
        self._c31_y = y
        return y

    # -----------------------------------------------------------------------
    # Private: TR4 loading — Ebers-Moll Newton-Raphson solver
    # -----------------------------------------------------------------------
    def _solve_env_ebers_moll(self, G_fixed: float, I_fixed: float) -> float:
        """
        Resolve V_env and the TR4 operating point via Ebers-Moll NR.
        Called by the ENV root (_ParallelNLRoot) during compute(); stores
        (V_env, gm, r_pi) in self._env_solution and returns the junction
        voltage −V_env (pywdf sign).

        Equation to solve each sample (Werner 2015 §IV, single-port NL):
            f(V) = V · G_fixed + I_fixed + Is_b · (exp(V/Vt) − 1) = 0
            f'(V) = G_fixed + Is_b · exp(V/Vt) / Vt          > 0  always

        G_fixed, I_fixed come from the subtree waves a_vals collected by the
        root in this sample.

        Monotonicity of f' guarantees a unique root and global convergence of
        Newton-Raphson from any starting point.  In practice 3-5 iterations
        suffice for convergence to 1 pV.

        Stored in self._env_solution
        ----------------------------
        V_env  : float   ENV node voltage [V]
        gm     : float   TR4 transconductance hFE·Is_b·exp(V/Vt)/Vt  [A/V]
        r_pi   : float   small-signal base resistance hFE·Vt/Ic  [Ω]
        """
        # Newton-Raphson: start from previous V_env for fast convergence
        V = self._V_env
        Vt = self.Vt
        Is_b = self.Is_b
        for _ in range(20):
            ex  = np.exp(np.clip(V / Vt, -40.0, 40.0))
            Ib  = Is_b * (ex - 1.0)
            f   = V * G_fixed + I_fixed + Ib
            fp  = G_fixed + Is_b * ex / Vt
            dV  = -f / fp
            V  += dV
            if abs(dV) < 1e-12:
                break

        # Collector current and small-signal parameters
        ex_sol = np.exp(np.clip(V / Vt, -40.0, 40.0))
        Ic     = self.hFE * Is_b * ex_sol
        Ic     = max(Ic, 1e-15)
        gm     = Ic / Vt
        r_pi   = float(np.clip(self.hFE * Vt / Ic, 500.0, 10e6))

        self._env_solution = (V, gm, r_pi)
        return -V

    # -----------------------------------------------------------------------
    # Private: TR4 collector step (sub-circuit B)
    # -----------------------------------------------------------------------
    def _step_tr4(self, V_env: float, gm: float, noise_i: float) -> float:
        """
        Advance TR4 collector one sample.  Returns V_col [V].

        Parameters
        ----------
        V_env : float   ENV node voltage (not used; r_base comes from gm).
        gm    : float   Transconductance from Ebers-Moll NR solve [A/V].
        noise_i : float  B1 current into the noise node N [A].
        """
        # r_base = r_pi || R52, with r_pi = hFE·Vt/Ic = hFE/gm
        if gm > 1e-12:
            r_pi   = self.hFE / gm
            r_base = r_pi * self.R52 / (r_pi + self.R52)
        else:
            r_base = self.R52

        # C29 — noise coupling, first-order bilinear IIR.
        # Seen from C29, the Norton source (noise_i, VR8U) is a Thevenin
        # source (noise_i·VR8U, VR8U):
        #   v_b = noise_i·VR8U · r_base/(VR8U + r_base) · H(s)
        #   H(s) = sτ / (1 + sτ),  τ = C29·(VR8U + r_base)
        # Bilinear discretisation (α = 2fs·τ/(1+2fs·τ), pole p = 2α−1):
        #   H(z) = α·(1 − z⁻¹) / (1 − (2α−1)·z⁻¹)
        # Direct form I:  y[n] = α·(x[n]−x[n-1]) + (2α−1)·y[n-1]
        # The load of C28 (snare side) on node N is not modelled.
        r_ser        = self.VR8U + r_base
        tau_c29      = self.C29 * r_ser
        k            = 2.0 * self.fs * tau_c29       # bilinear constant
        alpha_hp     = k / (1.0 + k)
        v_th         = noise_i * self.VR8U
        x_new        = v_th - self._v_base_x
        self._v_hp   = alpha_hp * x_new + (2.0 * alpha_hp - 1.0) * self._v_hp
        self._v_base_x = v_th
        self._v_base = self._v_hp * r_base / r_ser    # noise voltage at the base

        self._tr4_src.set_voltage(gm * self._v_base * self.R_SRC_TR4)
        self._tr4_node.compute()
        return self._vr9l.wave_to_voltage()   # node voltage, see _build_tr4_node

    # -----------------------------------------------------------------------
    # Private: ENV node construction
    # -----------------------------------------------------------------------
    def _build_env_node(self):
        """
        ENV = _ParallelNLRoot([ser_lc, ser_sc, rvsrc_mr_port, r52]), with the
        TR4 base (Shockley) at the root.
        """
        # LC branch
        self._rvsrc_lc = ResistiveVoltageSource(self.R_OFF)
        self._c23      = Capacitor(self.C23, self.fs)
        self._r47      = Resistor(self.R47)
        ser_lc = SeriesAdaptor(
            ParallelAdaptor(self._rvsrc_lc, self._c23),
            self._r47)

        # SC branch (source on GND side → inverted voltage)
        self._rvsrc_sc = ResistiveVoltageSource(self.R_OFF)
        self._r42      = Resistor(self.R42)
        self._c24      = Capacitor(self.C24, self.fs)
        self._r44      = Resistor(self.R44)
        ser_sc = SeriesAdaptor(
            self._r44,
            ParallelAdaptor(self._c24, SeriesAdaptor(self._r42, self._rvsrc_sc)))

        # MR → ENV Thevenin (one-sample delayed C26)
        # Thevenin seen by ENV looking into the MR branch:
        #   R_th = R46  (R46 connects n_MR to ENV; R45 is on the D9 side of C26)
        # When D9 conducts the path is R_src→R43→C25→D9→R45→n_MR→R46→ENV,
        # but C26 sits at n_MR and separates R45 from R46 in the Thevenin
        # seen from ENV.  R45 therefore does not appear in R_th.
        self._rvsrc_mr_port = ResistiveVoltageSource(self.R46)
        self._rvsrc_mr_port.set_voltage(0.0)

        # ENV pull-down
        self._r52 = Resistor(self.R52)

        ports = [ser_lc, ser_sc, self._rvsrc_mr_port, self._r52]
        self._env_solution = (0.0, 0.0, self.R_PI_MAX)
        self._root_env = _ParallelNLRoot(ports, self._solve_env_ebers_moll)

    # -----------------------------------------------------------------------
    # Private: maracas branch (sub-circuit M)
    # -----------------------------------------------------------------------
    def _build_mr_tree(self):
        """
        Maracas branch as an R-type junction with D8 and D9 at the root
        (pe78.wdf_nlroot.RTypeNLRoot). Nodes: 0 S (M253, R43, C25),
        1 A (C25, D8, D9), 2 K (D9, R45), 3 n_MR (R45, C26, ENV branch).
        The ENV branch is a resistive source R46 + R_th with the open-circuit
        voltage of ENV, set every sample in process_sample (step 5).
        """
        self._m253_mr = ResistiveVoltageSource(self.R_OFF)
        self._r43     = Resistor(self.R43)
        self._c25     = Capacitor(self.C25, self.fs)
        self._r45     = Resistor(self.R45)
        self._c26     = Capacitor(self.C26, self.fs)
        R_env_th_init = 1.0 / (
            1.0 / self.R47
            + 1.0 / (self.R42 + self.R44)
            + 1.0 / self.R52
        )
        self._env_mr = ResistiveVoltageSource(self.R46 + R_env_th_init)
        self._env_mr.set_voltage(0.0)
        S, A, K, M = range(4)
        nvt = self.D_N * self.VT_27C
        self._root_mr = RTypeNLRoot(
            4,
            [(self._m253_mr, S, -1), (self._r43, S, -1), (self._c25, S, A),
             (self._r45, K, M), (self._c26, M, -1), (self._env_mr, M, -1)],
            [(A, -1, diode_law(self.D_IS, nvt, reverse=True)),    # D8
             (A, K, diode_law(self.D_IS, nvt))])                  # D9

    # -----------------------------------------------------------------------
    # Private: TR4 collector node (sub-circuit B)
    # -----------------------------------------------------------------------
    def _build_tr4_node(self):
        """
        TR4 collector = RootRTypeAdaptor([l2, vr9l, tr4_src])
        All ports shunt to AC-GND (+12 V bypassed).
        R54 is the internal resistance of tr4_src, not a separate port.
        V_col is read from the VR9L port (parallel junction: every port has
        the node voltage).  A passive port is used on purpose: with the
        original pywdf (root R-type compute order, E1) the source port
        voltage is wrong, while passive ports are correct.
        """
        self._l2      = Inductor(self.L2_val, self.fs)
        self._vr9l    = Resistor(self.VR9L)
        self._tr4_src = ResistiveVoltageSource(self.R_SRC_TR4)
        self._tr4_src.set_voltage(0.0)

        self._tr4_node = RootRTypeAdaptor(
            [self._l2, self._vr9l, self._tr4_src],
            _par_scatter)

    # -----------------------------------------------------------------------
    # Private: C31 IIR coefficients
    # -----------------------------------------------------------------------
    def _make_c31_coeffs(self):
        """
        Bilinear IIR coefficients for the output high-pass filter.

        Physical path: V_col → [VR9L + R55] → C31 → Rload → GND
        Transfer function:
            H(s) = s·τ_z / (1 + s·τ_p)
            τ_z = C31·Rload          (zero)
            τ_p = C31·(VR9L+R55+Rload)  (pole)

        Returns (b, a) as length-2 arrays for the direct-form I recursion:
            y[n] = b[0]·x[n] + b[1]·x[n-1] − a[1]·y[n-1]
        """
        Rs    = self.VR9L + self.R55
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