#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
M253 trigger outputs as seen by the voices (P4, decision D11)
=============================================================
In ltspice/Practical-Electronics-1978-01_Pagina_24_Fig6.asc each output is a
5.2 V pulse through a 1N4148 (D1-D4: the M253 output stage, user
2026-11-18). Every voice (pe78/noise_voices.py, pe78/twint.py) models it as
a switched resistive source; the voice input is a gate (above 0.5 V: on).

    on:  V_ON behind R_ON. 5.2 V minus the 1N4148 drop at the end of the
         charge (LTspice B1 = 0: C23 at 4.84-4.86 V). R_ON = 350 Ohm as
         before P4 (C23 charged in about 0.1 ms; through the diode LTspice
         takes a few us; a smaller R_ON rings with the bilinear rule).
    off: 0 V behind R_OFF: the diode is reverse biased and the output is
         open (reverse current of the standard.dio 1N4148, 2.5 nA,
         neglected).

Until P4: 4.5 V on, 10 MOhm off. The lower charge of C23 and its discharge
through the 10 MOhm cut the long cymbal: ENV > 0.5 V for 0.654 s after the
LC gate, LTspice 0.804 s (tools/ltspice_compare.py).
"""

V_ON = 4.85       # [V]
R_ON = 350.0      # [Ohm]
R_OFF = 1e12      # [Ohm] open
