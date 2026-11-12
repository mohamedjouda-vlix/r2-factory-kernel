"""
R2 kernel: a direct implementation of the stochastic discrete-event factory
specification (augmented state X = (S, R), enabled-event map Gamma,
deterministic inter-event flow phi_s, pre-jump transition kernel K_theta,
explicit simultaneous-event resolution contract R_tie, and marked
event/job observations N = sum_n delta_(tau_n, M_n)).

The engine is model-agnostic: a Model supplies Gamma, the clock-setting
distributions, the jump map, and the tie rule.  The engine only executes
the hierarchy  Flow -> Event detection -> Jump/reset.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

INF = math.inf


# ---------------------------------------------------------------------------
# Clock-setting distributions (parameterised by mean and SCV)
# ---------------------------------------------------------------------------
@dataclass
class Dist:
    """Positive random variable given by its mean and squared coefficient
    of variation (SCV).  SCV = 0 -> deterministic; otherwise Gamma with
    shape 1/SCV (SCV = 1 is exponential, SCV = 1/k is Erlang-k)."""
    mean: float
    scv: float

    def sample(self, rng: np.random.Generator) -> float:
        if self.scv == 0.0:
            return self.mean
        k = 1.0 / self.scv
        return rng.gamma(k, self.mean / k)


# ---------------------------------------------------------------------------
# Observation layer: marked point process and integrals
# ---------------------------------------------------------------------------
@dataclass
class Observer:
    """Accumulates time integrals of state functionals and records marks.

    warmup: observations before this time are discarded (initial-transient
    deletion); the path itself is still generated from t = 0.
    """
    warmup: float = 0.0
    t_obs: float = 0.0                       # observed time after warm-up
    integrals: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)  # (tau, event, payload)

    def flow(self, t0: float, dt: float, functionals: dict):
        # integrate piecewise-constant functionals over [t0, t0+dt) ∩ [warmup, ∞)
        lo = max(t0, self.warmup)
        hi = t0 + dt
        if hi <= lo:
            return
        span = hi - lo
        self.t_obs += span
        for k, v in functionals.items():
            self.integrals[k] = self.integrals.get(k, 0.0) + v * span

    def mark(self, tau: float, ev: str, payload):
        self.marks.append((tau, ev, payload))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class R2Engine:
    """Executes a Model as a GSMP with explicit tie resolution.

    Model interface
    ---------------
    events            : tuple of event names (the set E)
    initial_state()   : physical state S_0
    enabled(S)        : Gamma(S), a set of event names
    new_clock(e, S, rng) : clock-setting draw for event e newly enabled in S
    jump(S, e, tau, obs, rng) -> S'
                      : one-event transition kernel applied at the
                        pre-jump state; may record marks on obs
    tie_order(E_star) : R_tie, a deterministic total order on simultaneous
                        events (returns a list)
    functionals(S)    : dict of state functionals to integrate (e.g. WIP)
    reset_on_jump(e)  : events whose clocks must be redrawn after the jump
                        even if they stay enabled (e.g. the event that fired)
    """

    def __init__(self, model, rng: np.random.Generator, tie_tol: float = 0.0):
        self.m = model
        self.rng = rng
        self.tie_tol = tie_tol

    def run(self, horizon: float, obs: Observer, max_events: int | None = None):
        m, rng = self.m, self.rng
        S = m.initial_state()
        R = {e: INF for e in m.events}
        for e in m.enabled(S):
            R[e] = m.new_clock(e, S, rng)
        tau = 0.0
        n = 0
        while True:
            en = m.enabled(S)
            # --- event detection: Delta tau_{n+1} = min_{e in Gamma(S)} R_e
            dt = min((R[e] for e in en), default=INF)
            if dt == INF:
                obs.flow(tau, horizon - tau, m.functionals(S))
                break
            if tau + dt > horizon:
                obs.flow(tau, horizon - tau, m.functionals(S))
                break
            E_star = [e for e in en if R[e] - dt <= self.tie_tol]
            # --- deterministic flow phi_s on [tau, tau + dt)
            obs.flow(tau, dt, m.functionals(S))
            for e in en:
                R[e] -= dt
            tau += dt
            # --- jump: apply R_tie, then the one-event kernel sequentially
            order = m.tie_order(E_star) if len(E_star) > 1 else E_star
            for e in order:
                if e not in m.enabled(S):      # may be disabled by an earlier
                    continue                    # event in the same instant
                before = m.enabled(S)
                S = m.jump(S, e, tau, obs, rng)
                after = m.enabled(S)
                for f in m.events:
                    if f not in after:
                        R[f] = INF
                    elif f not in before or f == e or f in m.reset_on_jump(e):
                        R[f] = m.new_clock(f, S, rng)
            n += 1
            if max_events is not None and n >= max_events:
                break
        return tau


# ---------------------------------------------------------------------------
# Model 1: single-server FIFO queue (G/G/1, optionally finite capacity K)
# ---------------------------------------------------------------------------
class SingleServer:
    events = ("arr", "dep")

    def __init__(self, ia: Dist, sv: Dist, K: float = INF, tie="dep_first",
                 track_dist: int = 0):
        self.ia, self.sv, self.K, self.tie = ia, sv, K, tie
        self.track_dist = track_dist   # >0: integrate 1{N=n}, n < track_dist
        self.next_id = 0

    def initial_state(self):
        return {"q": deque(), "lost": 0}

    def enabled(self, S):
        return {"arr", "dep"} if S["q"] else {"arr"}

    def new_clock(self, e, S, rng):
        return self.ia.sample(rng) if e == "arr" else self.sv.sample(rng)

    def reset_on_jump(self, e):
        return ()

    def jump(self, S, e, tau, obs, rng):
        q = S["q"]
        if e == "arr":
            jid = self.next_id
            self.next_id += 1
            # Palm observation: what the arriving job sees (before admission)
            if tau >= obs.warmup:
                obs.mark(tau, "seen", len(q))
            if len(q) >= self.K:
                S["lost"] += 1
                if tau >= obs.warmup:
                    obs.mark(tau, "loss", jid)
                return S
            q.append((jid, tau))
            if tau >= obs.warmup:
                obs.mark(tau, "arr", jid)
        else:  # departure of head-of-line job (FIFO matching contract)
            jid, a = q.popleft()
            obs.mark(tau, "dep", (jid, a))
        return S

    def tie_order(self, E_star):
        pri = ["dep", "arr"] if self.tie == "dep_first" else ["arr", "dep"]
        return sorted(E_star, key=pri.index)

    def functionals(self, S):
        n = len(S["q"])
        f = {"N": n, "busy": 1.0 if n else 0.0}
        if self.track_dist and n < self.track_dist:
            f[n] = 1.0
        return f


# ---------------------------------------------------------------------------
# Model 2: closed CONWIP serial line of M single-server FIFO stations
# ---------------------------------------------------------------------------
class ConwipLine:
    def __init__(self, procs: list[Dist], w: int):
        self.procs, self.w, self.M = procs, w, len(procs)
        self.events = tuple(f"d{k}" for k in range(self.M))

    def initial_state(self):
        # all w jobs released into station 0 at t = 0
        return {"q": [deque((j, 0.0) for j in range(self.w))]
                + [deque() for _ in range(self.M - 1)]}

    def enabled(self, S):
        return {f"d{k}" for k in range(self.M) if S["q"][k]}

    def new_clock(self, e, S, rng):
        return self.procs[int(e[1:])].sample(rng)

    def reset_on_jump(self, e):
        return ()

    def jump(self, S, e, tau, obs, rng):
        k = int(e[1:])
        jid, rel = S["q"][k].popleft()
        if k + 1 < self.M:
            S["q"][k + 1].append((jid, rel))
        else:
            # completion; CONWIP card returns and a new job is released
            if rel >= obs.warmup:
                obs.mark(tau, "out", tau - rel)
            elif tau >= obs.warmup:
                obs.mark(tau, "out_unmatched", None)
            S["q"][0].append((jid, tau))
        return S

    def tie_order(self, E_star):
        # downstream first (frees capacity before upstream pushes)
        return sorted(E_star, key=lambda e: -int(e[1:]))

    def functionals(self, S):
        return {"WIP": sum(len(x) for x in S["q"])}
