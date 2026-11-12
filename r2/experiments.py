"""Numerical experiments for the R2 kernel paper.  Produces results.json."""
import json, math, sys, time
from multiprocessing import Pool
import numpy as np
from r2kernel import Dist, Observer, R2Engine, SingleServer, ConwipLine

BASE_SEED = 20261111


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    from scipy.stats import t
    hw = t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return float(x.mean()), float(hw)


# ---------------------------------------------------------------- E1: M/M/1
def run_mm1(args):
    rho, rep, T, W = args
    rng = np.random.default_rng([BASE_SEED, 1, int(rho * 100), rep])
    m = SingleServer(Dist(1 / rho, 1.0), Dist(1.0, 1.0), track_dist=60)
    o = Observer(warmup=W)
    R2Engine(m, rng).run(T, o)
    seen = np.array([p for _, e, p in o.marks if e == "seen"])
    C = np.array([tau - p[1] for tau, e, p in o.marks
                  if e == "dep" and p[1] >= W])
    pi_time = np.array([o.integrals.get(n, 0.0) / o.t_obs for n in range(60)])
    pi_palm = np.bincount(seen, minlength=60)[:60] / len(seen)
    # sojourn-time quantiles and KS distance to Exp(mu - lambda)
    th = 1.0 - rho
    Cs = np.sort(C)
    F_emp = np.arange(1, len(Cs) + 1) / len(Cs)
    ks = float(np.max(np.abs(F_emp - (1 - np.exp(-th * Cs)))))
    return dict(rho=rho, rep=rep,
                N=o.integrals["N"] / o.t_obs,
                TH=len(C) / o.t_obs,
                CT=float(C.mean()),
                util=o.integrals["busy"] / o.t_obs,
                q50=float(np.quantile(C, .5)), q90=float(np.quantile(C, .9)),
                q99=float(np.quantile(C, .99)), ks=ks,
                pi_time=pi_time.tolist(), pi_palm=pi_palm.tolist(),
                nseen=int(len(seen)), njobs=int(len(C)))


# --------------------------------------------------------------- E2: G/G/1
def run_gg1(args):
    u, ca2, ce2, rep, T, W = args
    rng = np.random.default_rng([BASE_SEED, 2, int(u * 100), int(ca2 * 100),
                                 int(ce2 * 100), rep])
    m = SingleServer(Dist(1 / u, ca2), Dist(1.0, ce2))
    o = Observer(warmup=W)
    R2Engine(m, rng).run(T, o)
    C = np.array([tau - p[1] for tau, e, p in o.marks
                  if e == "dep" and p[1] >= W])
    return dict(u=u, ca2=ca2, ce2=ce2, rep=rep,
                Wq=float(C.mean() - 1.0), CT=float(C.mean()),
                N=o.integrals["N"] / o.t_obs, TH=len(C) / o.t_obs)


# --------------------------------------------------------- E3: CONWIP line
def run_conwip(args):
    w, scv, rep, T, W = args
    rng = np.random.default_rng([BASE_SEED, 3, w, int(scv * 100), rep])
    m = ConwipLine([Dist(2.0, scv) for _ in range(4)], w)
    o = Observer(warmup=W)
    R2Engine(m, rng).run(T, o)
    # count completions in the half-open window (W, T] to avoid double
    # counting an event that falls exactly on the warm-up boundary
    outs = [tau for tau, e, p in o.marks
            if e in ("out", "out_unmatched") and tau > W]
    cts = [p for _, e, p in o.marks if e == "out"]
    return dict(w=w, scv=scv, rep=rep, TH=len(outs) / o.t_obs,
                CT=float(np.mean(cts)), WIP=o.integrals["WIP"] / o.t_obs)


# ------------------------------------------------- E4: tie-resolution demo
def run_tie():
    out = []
    for K, label in ((math.inf, "D/D/1"), (1, "D/D/1/1")):
        for tie in ("dep_first", "arr_first"):
            m = SingleServer(Dist(1.0, 0.0), Dist(1.0, 0.0), K=K, tie=tie)
            o = Observer(warmup=10.0)
            R2Engine(m, np.random.default_rng(0)).run(10010.0, o)
            seen = [p for _, e, p in o.marks if e == "seen"]
            deps = [p for t, e, p in o.marks if e == "dep" and p[1] >= 10.0]
            C = [t - p[1] for t, e, p in o.marks
                 if e == "dep" and p[1] >= 10.0]
            nl = sum(1 for _, e, _ in o.marks if e == "loss")
            out.append(dict(model=label, tie=tie,
                            TH=len(deps) / o.t_obs,
                            N=o.integrals["N"] / o.t_obs,
                            seen=float(np.mean(seen)),
                            CT=float(np.mean(C)),
                            loss=nl / len(seen)))
    return out


# ---------------------------------------------- E5: unstable sample paths
def run_paths():
    res = {}
    for rho in (0.9, 1.0, 1.1):
        rng = np.random.default_rng([BASE_SEED, 5, int(rho * 100)])
        m = SingleServer(Dist(1 / rho, 1.0), Dist(1.0, 1.0))
        o = Observer(warmup=0.0)
        T = 20000.0
        R2Engine(m, rng).run(T, o)
        ev = sorted([(t, +1) for t, e, _ in o.marks if e == "arr"] +
                    [(t, -1) for t, e, _ in o.marks if e == "dep"])
        grid = np.linspace(0, T, 401)
        ts = np.array([t for t, _ in ev]); ds = np.cumsum([d for _, d in ev])
        idx = np.searchsorted(ts, grid, side="right") - 1
        N = np.where(idx >= 0, ds[np.maximum(idx, 0)], 0)
        res[str(rho)] = dict(t=grid.tolist(), N=N.tolist())
    # end-of-horizon N(T)/T over replications
    stats = {}
    for rho in (1.0, 1.1, 1.2):
        vals = []
        for rep in range(20):
            rng = np.random.default_rng([BASE_SEED, 6, int(rho * 100), rep])
            m = SingleServer(Dist(1 / rho, 1.0), Dist(1.0, 1.0))
            o = Observer(warmup=0.0)
            R2Engine(m, rng).run(20000.0, o)
            na = sum(1 for _, e, _ in o.marks if e == "arr")
            nd = sum(1 for _, e, _ in o.marks if e == "dep")
            vals.append((na - nd) / 20000.0)
        stats[str(rho)] = ci(vals)
    return res, stats


def main():
    t0 = time.time()
    res = {}
    with Pool(2) as pool:
        # E1
        jobs = [(r, k, 2.0e5, 1.0e4) for r in (0.5, 0.7, 0.8, 0.9)
                for k in range(40)]
        res["mm1"] = pool.map(run_mm1, jobs)
        print("E1 done", time.time() - t0, flush=True)
        # E2
        jobs = [(u, a, e, k, 1.5e5, 1.0e4)
                for u in (0.5, 0.8, 0.9) for a in (0.25, 1.0, 2.0)
                for e in (0.25, 1.0, 2.0) for k in range(20)]
        res["gg1"] = pool.map(run_gg1, jobs)
        print("E2 done", time.time() - t0, flush=True)
        # E3
        jobs = [(w, s, k, 2.0e4, 1.0e3) for w in range(1, 13)
                for s in (0.0, 0.25, 1.0) for k in range(10)]
        res["conwip"] = pool.map(run_conwip, jobs)
        print("E3 done", time.time() - t0, flush=True)
    res["tie"] = run_tie()
    res["paths"], res["drift"] = run_paths()
    print("all done", time.time() - t0, flush=True)
    json.dump(res, open("results.json", "w"))


if __name__ == "__main__":
    main()
