#PE 1978 Virtual Analog DIY e-Drums
#Copyright (c) 2026 Simone Pandolfi
#SPDX-License-Identifier: MIT OR Apache-2.0
"""
R-type junction with several nonlinear one-ports at the root (Werner 2015).

The linear part is a set of pywdf one-port leaves (Resistor, Capacitor,
ResistiveVoltageSource, ...) connected between the nodes of a junction.
Seen from the junction, leaf k between nodes (p, n) is a Thevenin source:

    v_k = b_k + R_k * i_k        (b_k reflected wave, i_k current p -> leaf)

The nonlinear elements are ports between nodes, each with a voltage-
controlled law i = f(v) (current from p to n through the element). Per
sample the root:

  1. collects b_k from the leaves (propagate_reflected_wave);
  2. writes the Thevenin seen by the ports, v_N = H*b - Z*i_N, with the
     junction matrices (MNA of the leaves, cached by their resistances);
  3. solves v_N - H*b + Z*f(v_N) = 0 by Newton-Raphson (warm start from the
     previous sample, step limit);
  4. computes the node voltages v = Kb*b - Ki*i_N and sends to every leaf
     the incident wave a_k = 2*v_k - b_k (accept_incident_wave).

KCL holds at every node and the ports satisfy their laws at the solution
(check with residuals()). Every node must have a path to GND through the
leaves (the junction matrix is inverted once per set of resistances).

Theory: the nonlinear elements that are coupled without delay (a WDF
capacitor is an instantaneous port resistance, not a delay) are grouped at
the single root of the structure as one multiport nonlinearity; the linear
part is an R-type adaptor derived from the MNA of the circuit; the root is
solved iteratively. Here the ports are voltage-controlled (unknowns: the port
voltages) and the Newton Jacobian is J = I + Z*G_N, G_N = d i_N / d v_N.
With one-port laws G_N is diagonal and the coupling between the ports is in
Z; a multiport law (e.g. a gate whose output current depends on its input and
output voltages, C12) adds off-diagonal blocks.
  - K. J. Werner, V. Nangia, J. O. Smith III, J. S. Abel, "Resolving Wave
    Digital Filters with Multiple/Multiport Nonlinearities", DAFx-15.
  - M. J. Olsen, K. J. Werner, J. O. Smith III, "Resolving Grouped
    Nonlinearities in Wave Digital Filters using Iterative Techniques",
    DAFx-16 (Newton at the root; diode pair and BJT case studies).
  - K. J. Werner, M. J. Olsen, M. Rest, J. D. Parker, "Generalizing Root
    Variable Choice in Wave Digital Filters with Grouped Nonlinearities",
    DAFx-17 (other root variables, e.g. the V_BE unknown of snare.py).

Used by noise_voices.py (maracas branch: D8, D9; bases root) and twint.py
(CD4011A gate of the twin-T, C12). The snare front end
(noise_voices.py, from C6) uses the same method with a current-controlled composite port
(TR5 + D7) and keeps its own solver.
"""

import numpy as np


def lexp(x, xm=80.0):
    """exp(x) and its derivative, continued linearly above xm (Newton safety)."""
    if x <= xm:
        e = np.exp(x)
        return e, e
    e = np.exp(xm)
    return e * (1.0 + x - xm), e


def diode_law(Is, nvt, reverse=False):
    """
    Shockley law of a diode on a port (p, n).

    reverse=False: anode at p, i = Is*(exp(v/nvt) - 1).
    reverse=True : anode at n, i = -Is*(exp(-v/nvt) - 1).
    Returns f(v) -> (i, di/dv); di/dv > 0 in both cases.
    """
    if reverse:
        def f(v):
            e, de = lexp(-v / nvt)
            return -Is * (e - 1.0), Is * de / nvt
    else:
        def f(v):
            e, de = lexp(v / nvt)
            return Is * (e - 1.0), Is * de / nvt
    return f


class RTypeNLRoot:
    """
    Parameters
    ----------
    n_nodes : int
        Junction nodes 0..n_nodes-1; -1 is GND.
    leaves : list of (leaf, p, n)
        pywdf one-port leaves and the nodes they connect.
    ports : list of (p, n, law)
        Nonlinear ports; law(v) -> (i, di/dv), v = v_p - v_n. law is None for
        a port that belongs to a multiport law.
    multiport : list of (indices, law), optional
        Laws that couple several ports: law(v[indices]) -> (i[indices],
        J), J[r, c] = d i[indices[r]] / d v[indices[c]]. A port with zero
        current (i = 0, a voltage probe) can be used as a controlling
        voltage, e.g. the input of a gate.
    step : float
        Largest Newton step on a port voltage [V].
    tol : float
        Newton tolerance on the port voltages [V].
    maxit : int
        Newton iterations per sample; RuntimeError if exceeded.

    Observable after compute()
    --------------------------
    v_nodes, v_ports, i_ports : numpy arrays
    iter_max : largest number of Newton iterations so far
    """

    def __init__(self, n_nodes, leaves, ports, step=0.2, tol=1e-10, maxit=200,
                 multiport=()):
        self.leaves = list(leaves)
        self.laws = [law for _, _, law in ports]
        self.multiport = [(np.asarray(idx, dtype=int), law) for idx, law in multiport]
        covered = [k for idx, _ in self.multiport for k in idx]
        for k, law in enumerate(self.laws):
            if (law is None) != (k in covered) or covered.count(k) > 1:
                raise ValueError(f"port {k}: give either a one-port law or one multiport law")
        self.step, self.tol, self.maxit = step, tol, maxit

        def inc(p, q):
            col = np.zeros(n_nodes)
            if p >= 0:
                col[p] = 1.0
            if q >= 0:
                col[q] = -1.0
            return col

        self.AL = np.array([inc(p, q) for _, p, q in self.leaves]).T
        self.AN = np.array([inc(p, q) for p, q, _ in ports]).T
        self._cache = {}
        n = len(ports)
        self.v_ports = np.zeros(n)
        self.i_ports = np.zeros(n)
        self.v_nodes = np.zeros(n_nodes)
        self._b = np.zeros(len(self.leaves))
        self.iter_max = 0

    def _matrices(self):
        R = np.array([leaf.Rp for leaf, _, _ in self.leaves])
        key = tuple(R)
        m = self._cache.get(key)
        if m is None:
            AL, AN = self.AL, self.AN
            Ginv = np.linalg.inv(AL @ np.diag(1.0 / R) @ AL.T)
            B = AL @ np.diag(1.0 / R)
            m = (Ginv @ B, Ginv @ AN, AN.T @ Ginv @ B, AN.T @ Ginv @ AN)
            self._cache[key] = m
        return m

    def _eval(self, v):
        """Port currents and Jacobian d i / d v of the laws (n x n)."""
        n = len(v)
        i = np.empty(n)
        g = np.zeros((n, n))
        for k, law in enumerate(self.laws):
            if law is not None:
                i[k], g[k, k] = law(v[k])
        for idx, law in self.multiport:
            i[idx], g[np.ix_(idx, idx)] = law(v[idx])
        return i, g

    def compute(self):
        b = np.array([leaf.propagate_reflected_wave() for leaf, _, _ in self.leaves])
        Kb, Ki, H, Z = self._matrices()
        h = H @ b
        v = self.v_ports.copy()
        eye = np.eye(len(v))
        for it in range(1, self.maxit + 1):
            i, g = self._eval(v)
            F = v - h + Z @ i
            J = eye + Z @ g
            dv = np.clip(np.linalg.solve(J, -F), -self.step, self.step)
            v = v + dv
            if np.max(np.abs(dv)) < self.tol:
                break
        else:
            raise RuntimeError("RTypeNLRoot: Newton did not converge")
        if it > self.iter_max:
            self.iter_max = it
        i, _ = self._eval(v)
        self.v_ports, self.i_ports = v, i
        self.v_nodes = Kb @ b - Ki @ i
        self._b = b
        vL = self.AL.T @ self.v_nodes
        for (leaf, _, _), vk, bk in zip(self.leaves, vL, b):
            leaf.accept_incident_wave(2.0 * vk - bk)

    def residuals(self):
        """
        Check of the last sample: (max |KCL| over the nodes [A],
        max |port voltage - (v_p - v_n)| [V], max |i - law(v)| [A]),
        from the leaf waves and the port solution.
        """
        iL = np.array([(leaf.a - leaf.b) / (2.0 * leaf.Rp) for leaf, _, _ in self.leaves])
        kcl = self.AL @ iL + self.AN @ self.i_ports
        dv = self.AN.T @ self.v_nodes - self.v_ports
        i, _ = self._eval(self.v_ports)
        return (float(np.max(np.abs(kcl))), float(np.max(np.abs(dv))),
                float(np.max(np.abs(i - self.i_ports))))
