#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE Tonal Drum Voices — Edge Detector + Twin-T Oscillator
=========================================================
Practical Electronics, January 1978 — Fig. 7.
Conga added from M252-M253 SGS-Ates TechnicalNotes 131

Voices: Bass Drum (IC4a), Hi Bongo (IC4b), Low Bongo (IC4c),
        Claves (IC4d), Conga (same structure, different capacitor values).

Structure:
  EdgeDetector: converts a digital gate into a positive trigger spike.
  TwinTDrum:    damped Twin-T oscillator around a CD4011A NAND gate used as
                an inverter (inputs tied), nonlinear (C12); VR4 sets the
                decay (see below).
  CD4011AGate:  square-law model of the gate (both stacks), see below.

Physical topology — 4 nodes, 9 linear WDF leaves, 2 nonlinear ports:
  n0 = NB_R    : midpoint of R11-R12 (C5 shunt to GND)
  n1 = NC_C    : midpoint of C6-C7  (R16 shunt to GND, trigger injection)
  n2 = NAND_IN : CD4011 pins 1+2
  n3 = NAND_OUT: CD4011 pin 3

  First T  (resistive branch): [NAND_IN]──[R11]──┬──[R12]──[NAND_OUT]
                                                 └──[C5]──[GND]

  Second T (capacitive branch):[NAND_IN]──[C6]──┬──[C7]──[NAND_OUT]
                                                └──[R16]──[GND]
                                                └──[trigger spike]

  Bias+Decay: [NAND_IN]──[R10 + VR4]──[NAND_OUT]

  Output:     [NAND_OUT]──[R41]──[instrument bus]

Leaf order (components list): 0:r11 1:r12 2:c5 3:c6 4:c7 5:r16 6:vtrig
7:r_bias 8:r_out. The leaves are joined by an R-type junction derived from
the MNA of the network (pe78/wdf_nlroot.py, RTypeNLRoot); the gate is a
multiport nonlinearity at the root: port n2-GND (input, no current) and port
n3-GND (output current, function of V_n2 and V_n3), solved by Newton per
sample (decision D3-A of the plan).

Gate model (CD4011AGate, decisions D1-A, D2-B of C12 and plan D1-A..D4-A):
  NAND with inputs tied: two NMOS in series pull n3 down, two PMOS in
  parallel pull it up. Each stack is one square-law (Shichman-Hodges)
  device with the same K: I = K/2 Vov^2 (1 + lam Vds) in saturation,
  K (Vov Vds - Vds^2/2)(1 + lam Vds) in triode, 0 below threshold.
  VDD = 5 V. Thresholds 1.5 V for N and P (RCA File No. 479, Fig. 7.7:
  supply current from Vi = 1.5 V to VDD - 1.5 V). K from the current at the
  switching point, about 0.5 mA at VDD = 5 V (Fig. 7.7): K = 0.93 mA/V^2;
  gm = 2.0 mS at the operating point. lam = 0.0292 1/V is tuned so that the
  bass drum self-oscillates above VR4 = 189.5 kOhm, as the former linear
  model (a = -40, r_o = 1 kOhm, decision D4-A): A0 = 73.4, r_out = 36.7 kOhm.
  At full drive (VGS = 5 V) the square law gives about 6 mA, against
  1.1/1.2 mA of the typical curves of Fig. 7.8/7.9: one square law cannot
  follow both figures; the operating point (Fig. 7.7) is preferred.
  Chip parameter vm (switching voltage, 1.5 ... 3.5 V, i.e. 30-70 % of VDD
  as in Fig. 7.4; default 2.5 V): the N and P thresholds move in opposite
  directions (VT0 +- d). The overdrive at the operating point stays 1 V, so
  gm, A0 and the VR4 threshold do not change with vm (189.5-189.6 kOhm);
  vm sets the DC level and the headroom towards the two rails, i.e. the
  asymmetry of the clipping.

DC operating point: C5, C6, C7 block DC and the gate input draws no current,
so V_n2 = V_n3 = V_dc with I_gate(V_dc, V_dc) = V_dc/R41 (R41 to GND in the
model): V_dc = vm - 3 mV. The capacitors start charged at the DC point
(no power-on transient); TonalDrum starts its C21 high-pass at V_dc.

VR4 semantics:
  r_bias = R10 + VR4  (total value in Ohm, not VR4 alone); R10 = 150 kOhm,
  VR4 = 0 ... ~470 kOhm, default 150 kOhm -> r_bias = 300 kOhm.
  The DC operating point does not depend on VR4. VR4 sets the AC feedback,
  i.e. the distance from self-oscillation: larger VR4 -> longer decay and
  slightly lower pitch. Bass drum, small signal: VR4 = 0 -> 105.6 Hz,
  tau 9.1 ms; 150 kOhm -> 86.9 Hz, 83 ms; self-oscillation above 189.5 kOhm
  (former linear model: 110 Hz, 90 Hz, same tau and threshold). Above the
  threshold the oscillation is limited by the rails and by the compression
  of the gate.
"""

import numpy as np
import pywdf.core.wdf as wdf
from scipy.optimize import brentq

from pe78.wdf_nlroot import RTypeNLRoot


# =============================================================================
#  EdgeDetector
# =============================================================================
class EdgeDetector:
    """
    Edge Detector — converts a digital gate to a positive trigger spike.
    (Practical Electronics, January 1978, Fig. 6 pulse shaper section)

    Topology:
      TRIG -> [Node_T1: VS || R14]──[C8]──> [Node_T2: || R15]──[R13]──>[D2]

      R14 (12 kOhm): pull-down, anchors Node_T1 to GND (M253 open-drain output)
      C8  (0.15 uF): AC differentiator;  tau1 = C8*R15 ~= 7 ms
      R15 (47 kOhm): shunt Node_T2 to GND
      R13 (27 kOhm): current limiter;    tau2 = C8*R13 ~= 4 ms
      D2  (1N4148):  half-wave rectifier, passes only positive spikes
                     (rising edge of the trigger gate); LTspice standard.dio
                     Is = 2.52 nA, N = 1.752, as D6-D9 in noise_voices.py
                     (drop 0.38/0.48/0.58 V at 0.01/0.1/1 mA)

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    r_vals : list of float
        [R14, R15, R13].
    c_val : float
        C8 capacitance in Farad.
    """

    # D2 1N4148, LTspice standard.dio (same constants as noise_voices.py)
    D_IS   = 2.52e-9      # [A]  saturation current
    D_N    = 1.752        # [-]  emission coefficient
    VT_27C = 25.852e-3    # [V]  kT/q at 27 C (LTspice)

    def __init__(self, fs, r_vals, c_val):
        r14 = float(r_vals[0])
        r15 = float(r_vals[1])
        r13 = float(r_vals[2])

        self.vs  = wdf.ResistiveVoltageSource(1e-3)  # near-ideal voltage source
        self.r14 = wdf.Resistor(r14)
        self.c8  = wdf.Capacitor(float(c_val), fs)
        self.r15 = wdf.Resistor(r15)
        self.r13 = wdf.Resistor(r13)

        # WDF tree: root = Diode (nonlinear element, open port)
        p_t1       = wdf.ParallelAdaptor(self.vs, self.r14)   # Node_T1: VS || R14
        s_c8       = wdf.SeriesAdaptor(p_t1, self.c8)         # series C8
        p_t2       = wdf.ParallelAdaptor(s_c8, self.r15)      # Node_T2: || R15
        self.s_r13 = wdf.SeriesAdaptor(p_t2, self.r13)        # series R13
        # D2 1N4148. pywdf Diode uses Vt*n_diodes as thermal voltage, so
        # n_diodes = D_N gives i = Is*(exp(v/(N*Vt)) - 1).
        self.diode = wdf.Diode(self.s_r13, Is=self.D_IS, Vt=self.VT_27C,
                               n_diodes=self.D_N)

    def process_sample(self, v_trigger: float) -> float:
        """
        Compute one sample of the edge detector output.

        Parameters
        ----------
        v_trigger : float
            Gate input voltage (e.g. 0 V or 4.5 V from M253).

        Returns
        -------
        float
            Voltage across R13 (~= 0 V at rest, positive spike on rising edge).
            Pass directly to TwinTDrum.process_sample().
        """
        self.vs.set_voltage(v_trigger)
        self.diode.accept_incident_wave(self.s_r13.propagate_reflected_wave())
        self.s_r13.accept_incident_wave(self.diode.propagate_reflected_wave())
        return self.r13.wave_to_voltage()


# =============================================================================
#  CD4011AGate
# =============================================================================
def _mos(k, lam, vov, vds):
    """
    Square-law MOS (Shichman-Hodges) with channel-length modulation.

    vov = V_GS - V_T, vds = V_DS (negative: drain and source swapped).
    Returns (i, di/dvov, di/dvds), i from drain to source.
    """
    if vds < 0.0:
        i, du, dw = _mos(k, lam, vov - vds, -vds)
        return -i, -du, du + dw
    if vov <= 0.0:
        return 0.0, 0.0, 0.0
    m = 1.0 + lam * vds
    if vds >= vov:                                   # saturation
        q = 0.5 * k * vov * vov
        return q * m, k * vov * m, q * lam
    q = k * (vov * vds - 0.5 * vds * vds)            # triode
    return q * m, k * vds * m, k * (vov - vds) * m + q * lam


class CD4011AGate:
    """
    CD4011A NAND with the inputs tied, as an inverter (see module docstring).

    Parameters
    ----------
    vm : float, optional
        Switching voltage V_in = V_out of the unloaded gate [V], 1.5 ... 3.5
        (chip parameter, default 2.5).
    """

    VDD   = 5.0         # [V]   supply
    VT0   = 1.5         # [V]   N and P threshold at vm = 2.5 V (Fig. 7.7)
    I_M   = 0.5e-3      # [A]   stack current at the switching point (Fig. 7.7)
    LAM   = 0.0292235   # [1/V] channel-length modulation, tuned (D4-A)
    VM_DEFAULT = 2.5    # [V]
    K = 2.0 * I_M / ((VDD / 2 - VT0) ** 2 * (1.0 + LAM * VDD / 2))   # [A/V^2]

    def __init__(self, vm=VM_DEFAULT):
        self.set_vm(vm)

    def set_vm(self, vm):
        """Move the N and P thresholds in opposite directions so that the
        unloaded switching voltage is vm."""
        vm = float(vm)
        if not 1.5 <= vm <= 3.5:
            raise ValueError(f"vm = {vm} V outside 1.5 ... 3.5 V")

        def f(d):
            self.vtn, self.vtp = self.VT0 + d, self.VT0 - d
            return self.current(vm, vm)[0]
        brentq(f, -1.45, 1.45, xtol=1e-15)
        self.vm = vm

    def current(self, v_in, v_out):
        """
        Output current from n3 to GND (NMOS minus PMOS) and its derivatives.

        Returns (i, di/dv_in, di/dv_out).
        """
        i_n, gn, on = _mos(self.K, self.LAM, v_in - self.vtn, v_out)
        i_p, gp, op = _mos(self.K, self.LAM, self.VDD - v_in - self.vtp, self.VDD - v_out)
        return i_n - i_p, gn + gp, on + op

    def law(self, v):
        """Multiport law for RTypeNLRoot: ports (n2-GND, n3-GND)."""
        i, g_in, g_out = self.current(v[0], v[1])
        return np.array([0.0, i]), np.array([[0.0, 0.0], [g_in, g_out]])

    def dc_point(self, r_load):
        """V_dc with V_in = V_out and the gate driving r_load to GND."""
        return brentq(lambda v: self.current(v, v)[0] + v / r_load,
                      0.0, self.VDD, xtol=1e-15)


# =============================================================================
#  TwinTDrum
# =============================================================================
class TwinTDrum:
    """
    Damped Twin-T oscillator around a CD4011A gate (Practical Electronics
    1978, Fig. 7 — Bass Drum and variants). See the module docstring.

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    r_params : list of float
        [R11, R12, R16].
    c_params : list of float
        [C5, C6, C7].
    vr4 : float
        VR4 potentiometer value in Ohm.
        Physical range: 0 ... ~470 kOhm.
    r_load : float
        R41 — output isolation resistor to instrument bus (390 kOhm typical).
    vm : float, optional
        Chip parameter: switching voltage of the gate [V], 1.5 ... 3.5
        (default 2.5).

    Observable after process_sample(): v_in, v_out (V_n2, V_n3) [V].
    """

    # Fixed feedback resistor R10, in series with VR4 (AC feedback; see module docstring)
    R10 = 150e3     # [Ohm]

    def __init__(self, fs, r_params, c_params, vr4, r_load,
                 vm=CD4011AGate.VM_DEFAULT):
        r11 = float(r_params[0])
        r12 = float(r_params[1])
        r16 = float(r_params[2])
        c5  = float(c_params[0])
        c6  = float(c_params[1])
        c7  = float(c_params[2])

        self._vr4 = float(vr4)
        self.gate = CD4011AGate(vm)

        # WDF components
        self.r11          = wdf.Resistor(r11)
        self.r12          = wdf.Resistor(r12)
        self.c5           = wdf.Capacitor(c5, fs)
        self.c6           = wdf.Capacitor(c6, fs)
        self.c7           = wdf.Capacitor(c7, fs)
        self.r16          = wdf.Resistor(r16)
        self.r_trigger_in = wdf.ResistiveVoltageSource(27e3)            # Z = R13
        self.r_bias       = wdf.Resistor(self.R10 + self._vr4)         # R10 + VR4
        self.r_out        = wdf.Resistor(float(r_load))                 # R41

        self.components = [
            self.r11,           # 0
            self.r12,           # 1
            self.c5,            # 2
            self.c6,            # 3
            self.c7,            # 4
            self.r16,           # 5
            self.r_trigger_in,  # 6
            self.r_bias,        # 7
            self.r_out,         # 8
        ]
        n0, n1, n2, n3, gnd = 0, 1, 2, 3, -1
        nodes = [(n2, n0), (n0, n3), (n0, gnd), (n2, n1), (n1, n3),
                 (n1, gnd), (n1, gnd), (n2, n3), (n3, gnd)]
        self.root = RTypeNLRoot(
            4, [(c, p, q) for c, (p, q) in zip(self.components, nodes)],
            ports=[(n2, gnd, None), (n3, gnd, None)],
            multiport=[([0, 1], self.gate.law)], tol=1e-12)
        self.reset()

    # -----------------------------------------------------------------------
    # Public setters — control rate
    # -----------------------------------------------------------------------
    def set_vr4(self, vr4: float):
        """
        Update VR4 at control rate (maps directly to the physical potentiometer).

        The feedback resistance becomes R10 + VR4. VR4 sets the AC feedback,
        not the DC operating point: larger VR4 -> longer decay, up to
        self-oscillation (bass drum: above about 189.5 kOhm), where the
        oscillation is limited by the gate (see module docstring).
        Physical range: 0 Ohm (VR4=0) ... ~470 kOhm.
        """
        self._vr4 = float(vr4)
        self.r_bias.set_resistance(self.R10 + self._vr4)

    def reset(self):
        """Put the circuit at its DC operating point (capacitors charged)."""
        v = self.gate.dc_point(self.r_out.Rp)
        self.v_dc = v
        for c, vc in ((self.c5, v), (self.c6, v), (self.c7, -v)):
            c.z = vc
            c.a = c.b = vc
        self.root.v_ports = np.array([v, v])
        self.r_out.a = self.r_out.b = v
        self.v_in = self.v_out = v

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------
    def process_sample(self, v_trigger: float) -> float:
        """
        Compute one audio sample.

        Parameters
        ----------
        v_trigger : float
            Output of EdgeDetector.process_sample() — positive spike,
            injected into NC_C (n1) via r_trigger_in (impedance = R13).

        Returns
        -------
        float
            Voltage at NAND_OUT (n3) = voltage across R41 (R41 to ground: the
            bus voltage is negligible against V_n3). Absolute voltage: at rest
            it is V_dc (about 2.5 V); TonalDrum removes it with C21.
        """
        self.r_trigger_in.set_voltage(v_trigger)
        self.root.compute()
        self.v_in, self.v_out = self.root.v_ports
        return self.v_out
