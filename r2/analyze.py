import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import ci

r = json.load(open("results.json"))
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
tex = {}

# ------------------------------------------------------------- E1 tables/fig
rows = []
for rho in (0.5, 0.7, 0.8, 0.9):
    rs = [x for x in r["mm1"] if x["rho"] == rho]
    f = lambda k: ci([x[k] for x in rs])
    N, CT, TH = f("N"), f("CT"), f("TH")
    pt = np.mean([x["pi_time"] for x in rs], 0)
    pp = np.mean([x["pi_palm"] for x in rs], 0)
    geo = (1 - rho) * rho ** np.arange(60)
    tv = 0.5 * np.abs(pt - geo).sum()
    tvp = 0.5 * np.abs(pp - pt).sum()
    lit = abs(N[0] - TH[0] * CT[0])
    q99 = f("q99")[0]
    rows.append((rho, N, rho / (1 - rho), CT, 1 / (1 - rho), TH, tv, tvp,
                 q99, math.log(100) / (1 - rho), f("ks")[0], lit,
                 int(np.mean([x["njobs"] for x in rs]))))
lines = []
for (rho, N, Nt, CT, Ct, TH, tv, tvp, q99, q99t, ks, lit, nj) in rows:
    lines.append(
        f"{rho:.1f} & {N[0]:.3f} $\\pm$ {N[1]:.3f} & {Nt:.3f} & "
        f"{CT[0]:.3f} $\\pm$ {CT[1]:.3f} & {Ct:.3f} & "
        f"{TH[0]:.4f} & {q99:.2f} & {q99t:.2f} & {ks:.4f} & "
        f"{tv:.4f} & {tvp:.4f} \\\\")
tex["tab_mm1"] = "\n".join(lines)

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
rho = 0.8
rs = [x for x in r["mm1"] if x["rho"] == rho]
pt = np.mean([x["pi_time"] for x in rs], 0)[:21]
pp = np.mean([x["pi_palm"] for x in rs], 0)[:21]
n = np.arange(21)
ax[0].plot(n, (1 - rho) * rho ** n, color=INK2, lw=1.2, ls="--",
           label=r"theory $(1-\rho)\rho^n$")
ax[0].plot(n, pt, "o", ms=4.5, mfc=C1, mec="white", mew=0.8,
           label=r"time-stationary $\hat\mu\{N=n\}$")
ax[0].plot(n, pp, "s", ms=3.2, mfc="none", mec=C2, mew=1.1,
           label=r"arrival-Palm $\hat{\mathbb{P}}^0_{\rm arr}\{N=n\}$")
ax[0].set_yscale("log"); ax[0].set_xlabel("$n$")
ax[0].set_ylabel("probability"); ax[0].legend(loc="lower left")
ax[0].set_title(r"(a) state laws, $\rho=0.8$", loc="left")
rr = np.linspace(0.45, 0.93, 100)
ax[1].plot(rr, rr / (1 - rr), color=C1, lw=1.2, label=r"$\rho/(1-\rho)$")
ax[1].plot(rr, 1 / (1 - rr), color=C2, lw=1.2, ls="--",
           label=r"$1/(\mu-\lambda)$")
for (rho, N, Nt, CT, *_rest) in rows:
    ax[1].errorbar(rho, N[0], yerr=N[1], fmt="o", ms=4.5, color=C1,
                   mec="white", mew=0.8)
    ax[1].errorbar(rho, CT[0], yerr=CT[1], fmt="s", ms=4, color=C2,
                   mec="white", mew=0.8)
ax[1].plot([], [], "o", color=C1, label=r"sim. $\mathbb{E}_\mu[N]$")
ax[1].plot([], [], "s", color=C2, label=r"sim. $\mathbb{E}^0_{\rm arr}[C]$")
ax[1].set_xlabel(r"$\rho$"); ax[1].legend(loc="upper left", ncol=1)
ax[1].set_title("(b) state route vs. Palm route", loc="left")
fig.tight_layout(); fig.savefig("../paper/fig_mm1.pdf")

# -------------------------------------------------------------- E2
lines = []
err = {}
for u in (0.5, 0.8, 0.9):
    for a in (0.25, 1.0, 2.0):
        cells = []
        for e in (0.25, 1.0, 2.0):
            rs = [x for x in r["gg1"] if x["u"] == u and x["ca2"] == a
                  and x["ce2"] == e]
            m, h = ci([x["Wq"] for x in rs])
            K = (a + e) / 2 * u / (1 - u)
            rel = 100 * (K - m) / m
            err[(u, a, e)] = (rel, 100 * h / m)
            cells.append(f"{m:.3f} $\\pm$ {h:.3f} & {K:.3f} & {rel:+.1f}")
        ulab = f"{u:.1f}" if a == 0.25 else ""
        lines.append(f"{ulab} & {a:g} & " + " & ".join(cells) + " \\\\")
        if a == 2.0 and u != 0.9:
            lines.append("\\midrule")
tex["tab_gg1"] = "\n".join(lines)

fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
us = (0.5, 0.8, 0.9)
for i, a in enumerate((0.25, 1.0, 2.0)):
    for e, col, mk in ((0.25, C1, "o"), (1.0, C2, "s"), (2.0, C3, "^")):
        y = [err[(u, a, e)][0] for u in us]
        yh = [err[(u, a, e)][1] for u in us]
        ax[i].errorbar(us, y, yerr=yh, color=col, marker=mk, ms=4.5,
                       mec="white", mew=0.8, capsize=0, lw=1.4,
                       label=f"$c_e^2={e:g}$")
    ax[i].axhline(0, color=INK2, lw=0.8)
    ax[i].set_title(f"$c_a^2={a:g}$" + (" (M/G/1: exact)" if a == 1 else ""),
                    loc="left")
    ax[i].set_xlabel("utilisation $u$"); ax[i].set_xticks(us)
ax[0].set_ylabel("VUT error in $\\mathbb{E}[W_q]$ (%)")
ax[2].legend(loc="upper right")
fig.tight_layout(); fig.savefig("../paper/fig_gg1.pdf")

# -------------------------------------------------------------- E3
ws = np.arange(1, 13)
T0, rb, W0 = 8.0, 0.5, 4.0
agg = {}
for s in (0.0, 0.25, 1.0):
    for w in ws:
        rs = [x for x in r["conwip"] if x["w"] == w and x["scv"] == s]
        agg[(s, w)] = (ci([x["TH"] for x in rs]), ci([x["CT"] for x in rs]))
lines = []
for w in ws:
    bc = min(w / T0, rb); pwc = w / (W0 + w - 1) * rb
    c = [agg[(s, w)] for s in (0.0, 0.25, 1.0)]
    lines.append(
        f"{w} & {bc:.4f} & {c[0][0][0]:.4f} & {c[1][0][0]:.4f} $\\pm$ "
        f"{c[1][0][1]:.4f} & {c[2][0][0]:.4f} $\\pm$ {c[2][0][1]:.4f} & "
        f"{pwc:.4f} & {c[2][1][0]:.2f} & {T0 + (w - 1) / rb:.2f} \\\\")
tex["tab_conwip"] = "\n".join(lines)

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
wf = np.linspace(1, 12, 200)
ax[0].plot(wf, np.minimum(wf / T0, rb), color=INK, lw=1.1,
           label="Best Case (BC)")
ax[0].plot(wf, wf / (W0 + wf - 1) * rb, color=INK2, lw=1.1, ls="--",
           label="PWC")
ax[1].plot(wf, np.maximum(T0, wf / rb), color=INK, lw=1.1)
ax[1].plot(wf, T0 + (wf - 1) / rb, color=INK2, lw=1.1, ls="--")
for s, col, mk, lab in ((0.0, C1, "o", "sim. $c_e^2=0$"),
                        (0.25, C3, "^", "sim. $c_e^2=0.25$"),
                        (1.0, C2, "s", "sim. $c_e^2=1$")):
    ax[0].plot(ws, [agg[(s, w)][0][0] for w in ws], mk, color=col, ms=4.5,
               mec="white", mew=0.8, label=lab)
    ax[1].plot(ws, [agg[(s, w)][1][0] for w in ws], mk, color=col, ms=4.5,
               mec="white", mew=0.8)
ax[0].set_xlabel("CONWIP level $w$"); ax[0].set_ylabel("TH (jobs/h)")
ax[1].set_xlabel("CONWIP level $w$"); ax[1].set_ylabel("CT (h)")
ax[0].legend(loc="lower right")
ax[0].set_title("(a) throughput", loc="left")
ax[1].set_title("(b) cycle time", loc="left")
fig.tight_layout(); fig.savefig("../paper/fig_conwip.pdf")

# ------------------------------------------------------------- economics
g, h = 100.0, 1.5
fig, ax = plt.subplots(figsize=(3.3, 2.4))
econ = []
for s, col, mk, lab in ((0.0, C1, "o", "$c_e^2=0$"),
                        (0.25, C3, "^", "$c_e^2=0.25$"),
                        (1.0, C2, "s", "$c_e^2=1$")):
    J = np.array([g * agg[(s, w)][0][0] - h * w for w in ws])
    k = int(np.argmax(J))
    econ.append((s, int(ws[k]), float(J[k]), float(g * agg[(s, ws[k])][0][0])))
    ax.plot(ws, J, marker=mk, color=col, ms=4.5, mec="white", mew=0.8,
            label=lab)
    ax.plot(ws[k], J[k], marker="o", ms=10, mfc="none", mec=INK, mew=0.9)
ax.set_xlabel("CONWIP level $w$")
ax.set_ylabel(r"$J(w)=g\,\mathrm{TH}-h\,w$  (\$/h)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("../paper/fig_econ.pdf")
tex["econ"] = econ

# ------------------------------------------------------------- paths
fig, ax = plt.subplots(figsize=(3.3, 2.4))
for rho, col, ls in (("0.9", C1, "-"), ("1.0", C3, "-"), ("1.1", C2, "-")):
    p = r["paths"][rho]
    ax.plot(np.array(p["t"]) / 1000, p["N"], color=col, lw=1.2, ls=ls,
            label=fr"$\rho={rho}$")
tt = np.linspace(0, 20, 50)
ax.plot(tt, 100 * tt, color=INK2, lw=0.9, ls=":",
        label=r"$(\lambda-\mu)t$, $\rho=1.1$")
ax.set_xlabel(r"time $t$ ($\times 10^3$)"); ax.set_ylabel("$N(t)$")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("../paper/fig_paths.pdf")

tex["tie"] = r["tie"]
tex["drift"] = r["drift"]
json.dump(tex, open("tex_tables.json", "w"), indent=1)
print(tex["econ"]); print(tex["drift"])
