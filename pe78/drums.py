#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 Drum Engine — Common interface
=====================================
Wrapper classes with a tick() interface for the tonal voices of the PE78 /
M253 drum machine. The noise voices (snare, cymbals, maracas) are one circuit
and one object: pe78.noise_voices.NoiseVoices (C15).

Voices:
  bd     Bass Drum          (AnalogDrum, tonal)
  hb     Hi Bongo           (AnalogDrum, tonal) -- also triggered by the snare (D6)
  lb     Low Bongo          (AnalogDrum, tonal)
  cl     Claves             (AnalogDrum, tonal)
  cd     Conga              (AnalogDrum, tonal)
  sd     Snare Drum         (NoiseVoices, noise-based)
  lc     Long Cymbal        (NoiseVoices, noise-based)
  sc     Short Cymbal       (NoiseVoices, noise-based)
  mr     Maracas            (NoiseVoices, noise-based)

Note on hb/sd trigger sharing:
  The snare output of the M253 (pin 5) also drives the Hi Bongo trigger line
  (pin 6) through D6: every SD hit triggers HB as well, an HB hit alone does
  not trigger the snare. R21 (12 kOhm, HB line to GND) is the path through
  which C22 of the snare discharges after the gate (see pe78/noise_voices.py). The
  HB voice is driven with the full gate level; the D6 drop (HB line about
  4.1 V) and the slow fall of the line after a snare gate are not modelled.

Instrument parameter dictionaries:
  All VR4 values represent the potentiometer alone (not R10+VR4).
  TwinTDrum adds R10=150 kOhm internally.
  Default VR4=150 kOhm -> r_bias = R10+VR4 = 300 kOhm.
"""

import numpy as np

from pe78.twint import CD4011AGate, EdgeDetector, TwinTDrum

NOISE_STD = 1.0     # [-]  unit-variance noise; the level (B1) is set in pe78/noise.py
from pe78.noise_voices import NoiseVoices  # noqa: E402,F401  (re-export)



# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------
class BaseWDFDrum:
    """Common interface for all PE78 drum voices."""

    def __init__(self, fs):
        self.fs = fs

    def process_sample(self, v_trig, noise_sample=0.0):
        """Advance the model by one sample. Override in subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  Tonal voices
# ---------------------------------------------------------------------------
class TonalDrum(BaseWDFDrum):
    """
    Tonal drum voice — EdgeDetector + TwinTDrum (damped sinusoidal oscillator).

    Used for: Bass Drum, Hi Bongo, Low Bongo, Claves, Conga.

    Output: voltage on the pre-amp input (VR10 = RLOAD), like the noise voices.
    The N_TONAL tonal stages are joined through their own R41 before the
    single C21, which feeds RLOAD. With one voice active and the others at AC
    ground (low-impedance gate outputs), the joined node is a Thevenin source
    (V_n3 / N_TONAL, R41 / N_TONAL):
        V_out = V_n3/N * RLOAD/(R41/N + RLOAD) * s*tau/(1 + s*tau)
        tau   = C21 * (R41/N + RLOAD)
    The output network is linear, so voices played together add. The load of
    the noise section on the bus (R55/C31) is applied outside, with
    bus_load_gains() (C16). V_n3 is absolute since C12 (DC level V_dc, about
    2.5 V); C21 blocks it and its state starts at V_dc / N_TONAL, so the
    output starts at 0 V without a transient.
    """

    C21     = 0.22e-6   # [F]   coupling capacitor, shared by the tonal voices
    RLOAD   = 10e3      # [Ohm] pre-amp input (VR10), as Rload in snare.py
    N_TONAL = 5         # tonal stages joined before C21 (bd, cd, hb, lb, cl)

    def __init__(self, fs, edge_r, edge_c, kick_r, kick_c, kick_vr4, kick_load,
                 vm=None):
        """
        Parameters
        ----------
        fs : int
            Sample rate in Hz.
        edge_r : list of float
            [R14, R15, R13, R16] for EdgeDetector.
        edge_c : float
            C8 capacitance in Farad for EdgeDetector.
        kick_r : list of float
            [R11, R12, R16] for TwinTDrum.
        kick_c : list of float
            [C5, C6, C7] for TwinTDrum.
        kick_vr4 : float
            VR4 potentiometer value in Ohm (TwinTDrum adds R10 internally).
        kick_load : float
            R41 output isolation resistor in Ohm.
        vm : float or None
            Chip parameter of the CD4011A gate: switching voltage [V],
            1.5 ... 3.5 (None: CD4011AGate.VM_DEFAULT = 2.5 V). See twint.py.
        """
        self.detector = EdgeDetector(fs, edge_r, edge_c)
        self.voice    = TwinTDrum(fs, kick_r, kick_c, kick_vr4, kick_load,
                                  vm=CD4011AGate.VM_DEFAULT if vm is None else vm)

        # C21 output coupling: bilinear HP (same form as snare.py C31)
        r_th  = float(kick_load) / self.N_TONAL
        tau_p = self.C21 * (r_th + self.RLOAD)
        tau_z = self.C21 * self.RLOAD
        k     = 2.0 * fs
        a0    = 1.0 + k * tau_p
        self._b_c21 = (k * tau_z / a0, -k * tau_z / a0)
        self._a1_c21 = (1.0 - k * tau_p) / a0
        self._c21_x = self.voice.v_dc / self.N_TONAL     # DC operating point
        self._c21_y = 0.0

    def tick(self, v_trigger, noise_sample=0.0):
        """
        Parameters
        ----------
        v_trigger : float
            Gate voltage (4.5 V = on, 0 V = off).
        noise_sample : float
            Unused; present for interface uniformity.

        Returns
        -------
        float
            Voltage on the pre-amp input [V] (see class docstring).
        """
        v_spike = self.detector.process_sample(v_trigger)
        x = self.voice.process_sample(v_spike) / self.N_TONAL    # V_th of the joined node
        y = (self._b_c21[0] * x + self._b_c21[1] * self._c21_x
             - self._a1_c21 * self._c21_y)
        self._c21_x = x
        self._c21_y = y
        return y

    def set_decay(self, vr4: float):
        """
        Set VR4 decay potentiometer at control rate.

        Parameters
        ----------
        vr4 : float
            VR4 value in Ohm (0 ... ~470 kOhm). Larger VR4 -> longer decay,
            up to self-oscillation (bass drum: above about 189.5 kOhm), limited
            by the CD4011A gate (see twint.py).
        """
        self.voice.set_vr4(vr4)


# ---------------------------------------------------------------------------
#  Instrument parameter dictionaries
#  kick_vr4 = VR4 alone (TwinTDrum adds R10=150 kOhm internally)
#  kick_vr4 default = 150 kOhm -> r_bias = 300 kOhm
# ---------------------------------------------------------------------------

BDO_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],  # R14, R15, R13, R16
    'edge_c':    150e-9,                         # C8
    'kick_r':    [68000, 68000, 10000],          # R11, R12, R16
    'kick_c':    [150e-9, 47e-9, 47e-9],         # C5, C6, C7
    'kick_vr4':  150000,                         # VR4 (r_bias = R10+VR4 = 300 kOhm)
    'kick_load': 390000,                         # R41 (390 kOhm, Fig. 7)
}

CONGA_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    56e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [56e-9, 18e-9, 18e-9],
    'kick_vr4':  150000,
    'kick_load': 390000,
}

HBONGO_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    33e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [33e-9, 10e-9, 10e-9],
    'kick_vr4':  150000,
    'kick_load': 390000,
}

LBONGO_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    47e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [47e-9, 15e-9, 15e-9],
    'kick_vr4':  150000,
    'kick_load': 390000,
}

CLAVES_PARAMS = {
    'edge_r':    [12000, 47000, 27000, 10000],
    'edge_c':    4.7e-9,
    'kick_r':    [68000, 68000, 10000],
    'kick_c':    [4.7e-9, 1.5e-9, 1.5e-9],
    'kick_vr4':  150000,
    'kick_load': 390000,
}


# ---------------------------------------------------------------------------
#  Output bus (C16)
# ---------------------------------------------------------------------------

def bus_load_gains(vr9_pos=None):
    """
    Mutual load of the two sections on the output bus (C16, decision D1-B).

    In the circuit the noise section (W -> R55 -> C31) and the tonal section
    (five R41 joined -> C21) drive the same node O, the pre-amp input VR10
    (RLOAD). TonalDrum and NoiseVoices each terminate on RLOAD alone; these
    mid-band gains (C21, C31 short, collector impedance neglected) give the
    voltage on O with the other section's branch connected:

        R_t = R41/N_TONAL                 tonal source resistance (78 kOhm)
        R_n = R55 + VR9U || VR9L          noise source resistance at O
        g_noise = [ (RL||R_t)/(R_n + RL||R_t) ] / [ RL/(R_n + RL) ]
        g_tonal = [ (RL||R_n)/(R_t + RL||R_n) ] / [ RL/(R_t + RL) ]

    Default VR9 (25k/195k): g_noise = 0.905 (-0.87 dB), g_tonal = 0.833
    (-1.59 dB). Against the nodal AC solution with C21, C31 and the
    collectors: within 0.06 dB from 200 Hz to 5 kHz, up to 0.6 dB at 50 Hz
    (tests/test_bus_load.py). TODO (D1-A, to re-evaluate): one bus node O
    inside the WDF of the noise section, with the tonal branch as a leaf.
    """
    nv = NoiseVoices
    pos = nv.VR9_POS_DEFAULT if vr9_pos is None else float(vr9_pos)
    u = max(pos * nv.VR9, 1.0)
    l = max((1.0 - pos) * nv.VR9, 1.0)
    rl = TonalDrum.RLOAD
    r_t = BDO_PARAMS['kick_load'] / TonalDrum.N_TONAL
    r_n = nv.R55 + u * l / (u + l)

    def par(a, b):
        return a * b / (a + b)
    g_noise = (par(rl, r_t) / (r_n + par(rl, r_t))) / (rl / (r_n + rl))
    g_tonal = (par(rl, r_n) / (r_t + par(rl, r_n))) / (rl / (r_t + rl))
    return g_noise, g_tonal
