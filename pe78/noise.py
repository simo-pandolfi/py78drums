#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
PE78 noise source (TR2 / B1) — single definition of the noise level
===================================================================
TR2 (base-emitter junction in avalanche, collector open) is the noise source
shared by the snare (via C28 to the TR3 base) and the cymbals (via C29 to the
TR4 base). There is no transistor model for this condition; the LTspice
reference (ltspice/Practical-Electronics-1978-01_Pagina_24_Fig6.asc) uses

    B1:  I = -white(2e6*time) * 4e-6

injected into the noise node N, which is pulled to +12 V by VR8U (the part
of the 100 kOhm trimmer VR8 between N and +12 V). With an ideal current
source the other part of VR8 is in series with it and has no effect.

white(x) is a random value in [-0.5, 0.5] that changes at every integer x,
with smoothed transitions. Below a few hundred kHz its spectrum is flat, with
one-sided density sigma_u * sqrt(2 / B1_RATE), sigma_u = B1_I_PP / sqrt(12),
whatever the smoothing.

Interface used by the models
  noise_sample : unit-variance white noise (sigma = 1), one value per sample.
  noise_amp(fs): B1 current per unit of noise_sample [A], chosen so that the
                 one-sided density equals the one of B1 for any fs.
  Seen from C28 (or C29), the Norton source (i, VR8U) is a Thevenin source
  (i * VR8U, VR8U). With r_base the AC resistance at the transistor base:
      v_b = i * VR8U * r_base / (VR8U + r_base) * s*tau / (1 + s*tau)
      tau = C * (VR8U + r_base)
  C28 and C29 share node N, so each voice also loads the other:
  NoiseVoices (pe78/noise_voices.py, C15) solves the TR3 and TR4 bases and
  node N together.
"""

import math

B1_RATE = 2e6       # [1/s] white(B1_RATE * time): one new value every 0.5 us
B1_I_PP = 4e-6      # [A]   scale of white(): current in [-2, 2] uA
VR8     = 100e3     # [Ohm] noise level trimmer (calibration)
VR8U    = 90e3      # [Ohm] VR8 between node N and +12 V (setting of the .asc)


def noise_density() -> float:
    """One-sided current density of B1 [A/sqrt(Hz)] (about 1.155e-9)."""
    return B1_I_PP / math.sqrt(12.0) * math.sqrt(2.0 / B1_RATE)


def noise_amp(fs: float) -> float:
    """
    B1 current per unit of a unit-variance noise sample at sample rate fs [A].

    sigma = noise_density() * sqrt(fs / 2): same density as B1 up to fs/2.
    Valid for fs much lower than B1_RATE.
    """
    return noise_density() * math.sqrt(fs / 2.0)
