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

Structure (one circuit per voice since C13):
  TwinTDrum:    edge detector (M253 output, R14, C8, R15, R13, D2) and damped
                Twin-T oscillator around a CD4011A NAND gate used as an
                inverter (inputs tied), nonlinear (C12); VR4 sets the decay.
  CD4011AGate:  square-law model of the gate (both stacks), see below.

Physical topology — 7 nodes, 13 linear WDF leaves, 3 nonlinear ports:
  n0 = NB_R    : midpoint of R11-R12 (C5 shunt to GND)
  n1 = NC_C    : midpoint of C6-C7  (R16 shunt to GND, cathode of D2)
  n2 = NAND_IN : CD4011 pins 1+2
  n3 = NAND_OUT: CD4011 pin 3
  t1           : M253 output (pin 4 for the bass drum), R14 to GND
  t2           : C8-R15-R13 node
  a            : anode of D2

  Edge detector:  [M253]──[t1]──[C8]──[t2]──[R13]──[a]──[D2 >|]──[n1]
                           R14 to GND   R15 to GND

  First T  (resistive branch): [NAND_IN]──[R11]──┬──[R12]──[NAND_OUT]
                                                 └──[C5]──[GND]

  Second T (capacitive branch):[NAND_IN]──[C6]──┬──[C7]──[NAND_OUT]
                                                └──[R16]──[GND]
                                                └──[D2 cathode]

  Bias+Decay: [NAND_IN]──[R10 + VR4]──[NAND_OUT]

  Output:     [NAND_OUT]──[R41]──[instrument bus]

M253 output: switched source of pe78/m253.py (D11), as in noise_voices.py:
on: V_ON = 4.85 V behind 350 Ohm, off: open (D1-D4 of the .asc). The
trigger input is a gate (above 0.5 V: on); until P4 its level was the
source voltage, with 10 MOhm off.
D2 1N4148: LTspice standard.dio, Is = 2.52 nA, N = 1.752 (C14; drop
0.38/0.48/0.58 V at 0.01/0.1/1 mA). D2 conducts on the rising edge of the
trigger and whenever n1 swings below t2 minus about one diode drop: the
detector network then loads the twin-T (damping that depends on the
amplitude). Before C13, D2 went to GND and the twin-T was fed by a source
equal to the voltage on R13 behind another 27 kOhm (R13 counted twice, no
coupling back), and R14 was shorted by a 1 mOhm source.

Twin-T leaf order (components list): 0:r11 1:r12 2:c5 3:c6 4:c7 5:r16
6:r_bias 7:r_out; detector leaves (detector list): 0:m253 1:r14 2:c8
3:r15 4:r13. All leaves are joined by one R-type junction derived from the
MNA of the network (pe78/wdf_nlroot.py, RTypeNLRoot); D2 (one-port, a-n1)
and the gate (multiport: n2-GND input with no current, n3-GND output current
as a function of V_n2 and V_n3) are solved together at the root by Newton
per sample (decision D3-A of the plan; the method of the plan for
nonlinearities coupled without delay).

Gate model (CD4011AGate, decisions D1-A, D2-B of C12 and plan D1-A..D4-A):
  NAND with inputs tied: two NMOS in series pull n3 down, two PMOS in
  parallel pull it up. Each stack is one square-law (Shichman-Hodges)
  device with the same K: I = K/2 Vov^2 (1 + lam Vds) in saturation,
  K (Vov Vds - Vds^2/2)(1 + lam Vds) in triode, 0 below threshold.
  VDD = 5 V. Thresholds 1.5 V for N and P (RCA File No. 479, Fig. 7.7:
  supply current from Vi = 1.5 V to VDD - 1.5 V). K from the current at the
  switching point, about 0.5 mA at VDD = 5 V (Fig. 7.7): K = 0.93 mA/V^2;
  gm = 2.0 mS at the operating point. lam = 0.0191 1/V is tuned so that the
  bass drum self-oscillates above VR4 = 189.5 kOhm, as the former linear
  model (a = -40, r_o = 1 kOhm, decision D4-A), on the circuit of C13 (n1
  loaded by R16 and D2 at 0 V; decision E2): A0 = 109.9, r_out = 55 kOhm.
  (C12 had lam = 0.0292, A0 = 73.4, tuned with the 27 kOhm load of the old
  trigger source on n1.)
  At full drive (VGS = 5 V) the square law gives about 6 mA, against
  1.1/1.2 mA of the typical curves of Fig. 7.8/7.9: one square law cannot
  follow both figures; the operating point (Fig. 7.7) is preferred.
  Chip parameter vm (switching voltage, 1.5 ... 3.5 V, i.e. 30-70 % of VDD
  as in Fig. 7.4; default 2.5 V): the N and P thresholds move in opposite
  directions (VT0 +- d). The overdrive at the operating point stays 1 V, so
  gm, A0 and the VR4 threshold do not change with vm (within 0.1 kOhm);
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
  slightly lower pitch. Bass drum, small signal: VR4 = 0 -> 92.8 Hz,
  tau 9.8 ms; 150 kOhm -> 77.8 Hz, 92.0 ms; self-oscillation above
  189.5 kOhm (former linear model with the 27 kOhm load on n1: 110 Hz/9.1 ms,
  90.3 Hz/82.7 ms, 189.5 kOhm). Thresholds of the other voices 188.4-191.2
  kOhm. Above the threshold the oscillation is
  limited by the rails, by the compression of the gate and, on the negative
  swings of n1, by D2.
"""

import numpy as np
import pywdf.core.wdf as wdf
from scipy.optimize import brentq

from pe78 import m253
from pe78.wdf_nlroot import RTypeNLRoot, diode_law


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
    LAM   = 0.0190638   # [1/V] channel-length modulation, tuned (D4-A, E2)
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
    One tonal voice of Fig. 7: edge detector and damped Twin-T oscillator
    around a CD4011A gate (Practical Electronics 1978 — Bass Drum and
    variants). See the module docstring.

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    edge_r : list of float
        [R14, R15, R13] (a fourth value, R16, is ignored: it is in r_params).
    edge_c : float
        C8 in Farad.
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
    d6 : bool, optional
        Hi Bongo only: add D6 from the snare output to the trigger line
        (node t1), driven through process_sample(..., line) (C13).

    Observable after process_sample(): v_in, v_out (V_n2, V_n3), v_n1,
    i_d2 (current of D2), i_d6 (with d6=True).
    """

    # Fixed feedback resistor R10, in series with VR4 (AC feedback; see module docstring)
    R10 = 150e3     # [Ohm]
    # M253 output (pe78/m253.py, as noise_voices.NoiseVoices)
    V_ON, R_ON, R_OFF = m253.V_ON, m253.R_ON, m253.R_OFF
    # D2 1N4148, LTspice standard.dio (same constants as noise_voices.py)
    D_IS   = 2.52e-9      # [A]  saturation current
    D_N    = 1.752        # [-]  emission coefficient
    VT_27C = 25.852e-3    # [V]  kT/q at 27 C (LTspice)

    # HB line driven by the snare (d6=True): idle Thevenin of the pin 5 node
    R_LINE_IDLE = 22e3    # [Ohm] about R48 + C22 with the M253 output off

    def __init__(self, fs, edge_r, edge_c, r_params, c_params, vr4, r_load,
                 vm=CD4011AGate.VM_DEFAULT, d6=False):
        r14, r15, r13 = (float(x) for x in edge_r[:3])
        r11 = float(r_params[0])
        r12 = float(r_params[1])
        r16 = float(r_params[2])
        c5  = float(c_params[0])
        c6  = float(c_params[1])
        c7  = float(c_params[2])

        self._vr4 = float(vr4)
        self._on = False
        self.gate = CD4011AGate(vm)

        # WDF leaves: edge detector
        self.m253 = wdf.ResistiveVoltageSource(self.R_OFF)
        self.r14  = wdf.Resistor(r14)
        self.c8   = wdf.Capacitor(float(edge_c), fs)
        self.r15  = wdf.Resistor(r15)
        self.r13  = wdf.Resistor(r13)
        # WDF leaves: twin-T
        self.r11    = wdf.Resistor(r11)
        self.r12    = wdf.Resistor(r12)
        self.c5     = wdf.Capacitor(c5, fs)
        self.c6     = wdf.Capacitor(c6, fs)
        self.c7     = wdf.Capacitor(c7, fs)
        self.r16    = wdf.Resistor(r16)
        self.r_bias = wdf.Resistor(self.R10 + self._vr4)         # R10 + VR4
        self.r_out  = wdf.Resistor(float(r_load))                 # R41

        self.components = [self.r11, self.r12, self.c5, self.c6, self.c7,
                           self.r16, self.r_bias, self.r_out]
        self.detector = [self.m253, self.r14, self.c8, self.r15, self.r13]
        n0, n1, n2, n3, t1, t2, a, s, gnd = 0, 1, 2, 3, 4, 5, 6, 7, -1
        nodes = [(n2, n0), (n0, n3), (n0, gnd), (n2, n1), (n1, n3),
                 (n1, gnd), (n2, n3), (n3, gnd),
                 (t1, gnd), (t1, gnd), (t1, t2), (t2, gnd), (t2, a)]
        leaves = self.components + self.detector
        d_law = diode_law(self.D_IS, self.D_N * self.VT_27C)
        ports = [(a, n1, d_law), (n2, gnd, None), (n3, gnd, None)]
        self.d6 = bool(d6)
        if self.d6:
            # Hi Bongo: the snare output (M253 pin 5) drives this line through
            # D6 (C13). Leaf v1src = Thevenin of the pin 5 / C22 node of the
            # snare front end (NoiseVoices.hb_line(), same sample), port D6.
            self.v1src = wdf.ResistiveVoltageSource(self.R_LINE_IDLE)
            leaves = leaves + [self.v1src]
            nodes = nodes + [(s, gnd)]
            ports = ports + [(s, t1, d_law)]
        self.root = RTypeNLRoot(
            8 if self.d6 else 7, [(c, p, q) for c, (p, q) in zip(leaves, nodes)],
            ports=ports, multiport=[([1, 2], self.gate.law)], tol=1e-12)
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
        """Put the circuit at its DC operating point (capacitors charged,
        M253 output off, D2 at 0 V)."""
        if self._on:
            self.m253.set_resistance(self.R_OFF)
            self._on = False
        self.m253.set_voltage(0.0)
        v = self.gate.dc_point(self.r_out.Rp)
        self.v_dc = v
        for c, vc in ((self.c5, v), (self.c6, v), (self.c7, -v), (self.c8, 0.0)):
            c.z = vc
            c.a = c.b = vc
        self.root.v_ports = np.array([0.0, v, v] + ([0.0] if self.d6 else []))
        if self.d6:
            self.v1src.set_resistance(self.R_LINE_IDLE)
            self.v1src.set_voltage(0.0)
        self.i_d6 = 0.0
        self.r_out.a = self.r_out.b = v
        self.v_in = self.v_out = v
        self.v_n1 = 0.0
        self.i_d2 = 0.0

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------
    def process_sample(self, v_trigger: float, line=None) -> float:
        """
        Compute one audio sample.

        Parameters
        ----------
        v_trigger : float
            M253 gate: above 0.5 V the output is on (V_ON behind R_ON),
            otherwise off (0 V behind R_OFF, open): pe78/m253.py.
        line : (float, float) or None
            Only with d6=True: (v1_oc, r_th1) of the snare front end, i.e.
            NoiseVoices.hb_line() after its tick of the same sample. None:
            snare idle (0 V behind R_LINE_IDLE).

        Returns
        -------
        float
            Voltage at NAND_OUT (n3) = voltage across R41 (R41 to ground: the
            bus voltage is negligible against V_n3). Absolute voltage: at rest
            it is V_dc (about 2.5 V); TonalDrum removes it with C21.
        """
        on = v_trigger > 0.5
        if on != self._on:
            self.m253.set_resistance(self.R_ON if on else self.R_OFF)
            self._on = on
        self.m253.set_voltage(self.V_ON if on else 0.0)
        if self.d6:
            v_oc, r_th = (0.0, self.R_LINE_IDLE) if line is None else line
            self.v1src.set_resistance(r_th)
            self.v1src.set_voltage(v_oc)
        self.root.compute()
        self.v_in, self.v_out = self.root.v_ports[1:3]
        self.v_n1 = self.root.v_nodes[1]
        self.i_d2 = self.root.i_ports[0]
        if self.d6:
            self.i_d6 = self.root.i_ports[3]
        return self.v_out
