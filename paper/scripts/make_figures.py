"""Figures for the arXiv version, generated from the released artifacts only.  No hand-entered numbers.

FIG_REACHABLE -- what the two intervention families actually reach: the exact slice, the real
    GPT-2 states, and the support gap that a naive comparison leaves between them.
FIG_REGIMES   -- the operator sets the exact onset d*; in a trained model exact blindness is
    gone and the evidence is merely ill-conditioned.
FIG_SPECTRUM  -- held-out response-ratio profiles per cell under the support-matched protocol,
    against the extreme-value same-family envelope.  (Appendix.)
FIG_MATCHED   -- the reverse direction is a support artifact, and what survives matching is
    predicted by the measured edge strength.
FIG_SCALES    -- IOI: single-site envelope, joint envelope and executed RMS T_C per capture (Section 7).

The spectra are HELD-OUT ORDER STATISTICS of the ten train-fitted generalized eigen-directions (optimised
directions), not the named generator of Section 5; the labels say so.

All three trained-model figures read the SUPPORT-MATCHED protocol, which is the primary one;
the superseded unmatched numbers live in matched_invariant.json under "frozen".
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.environ.get("E0_DIR", "/data/ai_models/explainability/release/e0")
INV = json.load(open(os.path.join(E0, "invariant.json")))


def _load(name):
    p = os.path.join(E0, name)
    return json.load(open(p)) if os.path.exists(p) else None

TW = 5.5   # NeurIPS \textwidth in inches; draw at final size so label sizes are honest
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


ORDER = ["GPT-2 IOI", "GPT-2 induction", "GPT-2 capital", "GPT-2 greater", "GPT-2 agreement",
         "Qwen3.5 IOI", "Qwen3.5 induction", "Qwen3.5 capital", "Qwen3-8B IOI", "Qwen3-8B induction"]
SHOW = {"GPT-2 greater": "GPT-2 greater-than", "Qwen3.5 IOI": "Qwen3.5-9B IOI",
        "Qwen3.5 induction": "Qwen3.5-9B induction", "Qwen3.5 capital": "Qwen3.5-9B capital"}


def fig_spectrum(path):
    """Held-out response-ratio profiles under the PRIMARY support-matched protocol.

    This is the raw evidence behind the headline: each thin line is one site pair's profile,
    frozen on train and re-scored on the disjoint complement, against the extreme-value
    same-family envelope.  Escaping at the low end is the theory-named blind direction.
    """
    M = _load("matched_invariant.json")
    if M:
        cells = {k: v["matched"] for k, v in M["cells"].items() if v.get("matched")}
        src = {k.replace(" ioi", " IOI"): v for k, v in cells.items()}
    else:
        src = INV
    keys = [k for k in ORDER if k in src] or sorted(src)
    ncol = 5 if len(keys) > 5 else len(keys)
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TW, 2.1 * nrow), sharey=True,
                             squeeze=False)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        d = src[k]
        spec = np.array(d["spectra"])              # [n_trials, p] held-out log10 ratios
        lo, hi = d["null_lo"], d["null_hi"]        # EXTREME-VALUE same-family envelope
        ax.axhspan(lo, hi, color="0.85", zorder=0)
        ax.axhline(0, color="0.55", lw=0.6, ls=":", zorder=1)
        n, p = spec.shape
        x = np.arange(1, p + 1)
        for i in range(n):
            ax.plot(x, np.sort(spec[i]), color="C0", alpha=0.35, lw=0.8, zorder=2)
        ax.plot(x, np.sort(spec, axis=1).mean(0), color="C3", lw=1.6, zorder=3)
        ax.set_title(f"{SHOW.get(k, k)}\nlow {d['below']}/{d['n']}\nreverse {d['above']}/{d['n']}",
                     fontsize=6.5)
        ax.set_xticks([1, p])
    for ax in axes[:, 0]:
        ax.set_ylabel(r"held-out $\log_{10}(E_S/E_J)$" + "\n(negative: S-weak)")
    fig.supxlabel("rank of held-out value among the ten train-fitted directions", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


def fig_regimes(path):
    """The three-regime bridge from the exact theory (S3) to the trained-model geometry (S4)."""
    import json as _j
    OD = {x["operator"]: x for x in _j.load(open(os.path.join(E0, "operator_degree.json")))}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TW, 2.3),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    names = ["affine", "rmsnorm", "softmax"]
    labels = ["affine: $d^*=2$", "RMSNorm: $d^*=5$", "softmax: $d^*=\\infty$"]
    for i, (nm, lab) in enumerate(zip(names, labels)):
        pd = OD[nm]["numeric"]["per_degree"]
        ds = sorted(int(d) for d in pd)
        ys = [max(pd[str(d)] if str(d) in pd else pd[d]) for d in ds]
        ax.semilogy(ds, np.maximum(ys, 1e-17), "o-", ms=3.5, lw=1.4, label=lab,
                    color=f"C{i}")
    ax.set_xticks(ds)
    ax.axhline(1e-13, color="0.6", lw=0.7, ls="--")
    ax.text(2.2, 2.5e-13, "reference level $10^{-13}$", fontsize=6.5, color="0.4", ha="left")
    ax.set_ylim(1e-17, 2.0)
    ax.set_xlabel("degree of the mechanism class", fontsize=7.5)
    ax.set_ylabel("SVD-candidate residual\n(relative, max over spans)", fontsize=7.5)
    ax.set_title("exact regime: the operator sets $d^*$", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", bbox_to_anchor=(1.04, 1.04),
              ncol=1, handlelength=0.8, labelspacing=0.2, borderpad=0.1)

    M = _load("matched_invariant.json")
    d = (M["cells"]["GPT-2 ioi"]["matched"] if M and "GPT-2 ioi" in M.get("cells", {})
         else INV.get("GPT-2 IOI"))
    spec = np.array(d["spectra"])
    lo, hi = d["null_lo"], d["null_hi"]
    ax2.axhspan(lo, hi, color="0.85", zorder=0)
    x = np.arange(1, spec.shape[1] + 1)
    for i in range(spec.shape[0]):
        ax2.plot(x, np.sort(spec[i]), color="C0", alpha=0.35, lw=0.8)
    ax2.plot(x, np.sort(spec, axis=1).mean(0), color="C3", lw=1.7)
    ax2.set_xticks([1, 5, 10])
    ax2.set_xlabel("rank of held-out value\n(ten train-fitted directions)", fontsize=7.5)
    ax2.set_ylabel(r"held-out $\log_{10}(E_S/E_J)$" + "\n(negative: lower single-site energy)", fontsize=7.5)
    ax2.set_title("trained model, tested cubic class:\nfinite evidence is ill-conditioned", fontsize=8)
    ax2.axhline(0, color="0.55", lw=0.6, ls=":")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(path)
    print("wrote", path)


def fig_reachable(path, capture="ioi_matched_gpt2_5_10_0.npz"):
    """What the two intervention families actually reach -- exact slice, and a trained model.

    This is the object the whole paper is about, and it is worth one look: under
    patch-and-recompute the experimenter chooses the intervention but the NETWORK decides which
    downstream configuration that choice reaches, so the single-site family traces a union of
    two curves while the joint family fills a region.  Panels (b) and (c) also show, in real
    GPT-2 states, the support gap that the frozen protocol left between the two families and
    what matching the interpolation grid does to it.
    """
    f = os.path.join(E0, capture)
    if not os.path.exists(f):
        print("skip fig_reachable: no", capture)
        return
    d = np.load(f)
    Zs, Zj, Zjm = d["Zs"], d["Zj"], d["Zjm"]
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.25))

    # (a) the exact affine slice
    ax = axes[0]
    a0, m, c = 0.35, 1.6, 0.2
    av = np.linspace(-2, 2, 200)
    ax.plot(av, m * av + c, color="C0", lw=1.6, label=r"patch $A$: $b$ recomputes")
    ax.plot(np.full(200, a0), np.linspace(-3.4, 3.4, 200), color="C0", lw=1.6, ls="--",
            label=r"patch $B$: $a$ pinned at $a_0$")
    A, B = np.meshgrid(np.linspace(-2, 2, 240), np.linspace(-3.4, 3.4, 240))
    ax.contourf(A, B, np.abs((A - a0) * (B - m * A - c)), levels=[0, 0.35],
                colors=["0.85"], zorder=0)
    ax.set_title("(a) exact affine slice", fontsize=7.5)
    ax.set_xlabel("$a$"); ax.set_ylabel("$b$")
    ax.set_ylim(-3.4, 6.0)
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [Patch(facecolor="0.85", label=r"$|g_R|$ small")], l + [r"$|g_R|$ small"], fontsize=6.5,
              frameon=False, loc="upper center", ncol=1, handlelength=1.6, borderaxespad=0.1)

    # (b), (c) the same trained-model states, against each joint family
    for ax, Zjx, ttl in ((axes[1], Zj, "(b) GPT-2: joint at $\\alpha=1$"),
                         (axes[2], Zjm, "(c) GPT-2: joint on the $\\alpha$ grid")):
        ax.scatter(Zjx[:, 0], Zjx[:, 1], s=3, color="C1", alpha=0.70, lw=0, zorder=2,
                   label="joint states")
        ax.scatter(Zs[:, 0], Zs[:, 1], s=3, color="C0", alpha=0.85, lw=0, zorder=3,
                   label="single-site states")
        r = np.ptp(Zs[:, 0]) / max(np.ptp(Zjx[:, 0]), 1e-30)
        ax.set_title(ttl, fontsize=7.5)
        ax.set_xlabel("$a$ (site $A$)", fontsize=7.5)
    axes[1].legend(fontsize=6.2, frameon=False, loc="upper left", markerscale=2.2, borderaxespad=0.1)
    axes[1].set_ylabel("$b$ (site $B$)", fontsize=7.5)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path)
    print("wrote", path)


def fig_matched(path):
    """Reverse blindness against the joint family's alpha grid -- the confound, within-trial."""
    M = _load("matched_invariant.json")
    if not M:
        print("skip fig_matched: no matched_invariant.json")
        return
    have = {k.replace(" ioi", " IOI"): k for k in M["cells"] if M["cells"][k].get("matched")}
    cells = [have[k] for k in ORDER if k in have]
    if not cells:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(TW, 2.6),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(cells)); w = 0.38
    fr = [100 * M["cells"][k]["frozen"]["above"] / M["cells"][k]["frozen"]["n"] for k in cells]
    mt = [100 * M["cells"][k]["matched"]["above"] / M["cells"][k]["matched"]["n"] for k in cells]
    sb = [100 * M["cells"][k]["matched"]["below"] / M["cells"][k]["matched"]["n"] for k in cells]
    ax.bar(x - w / 2, fr, w, color="C3", label="reverse end, joint at $\\alpha=1$")
    ax.bar(x + w / 2, mt, w, color="C0", label="reverse end, matched $\\alpha$ grid")
    ax.plot(x, sb, "k^", ms=5, label="low end, matched")
    ax.set_ylabel("% of captures escaping\nthe same-family envelope")
    ax.set_xticks(x)
    short = {"GPT-2": "G2", "Qwen3.5": "Q3.5", "Qwen3-8B": "Q3-8B"}
    lab = []
    for k in cells:
        mdl, task = k.split(" ", 1)
        lab.append(f"{short.get(mdl, mdl)} {task.replace('ioi', 'IOI').replace('greater', 'gt').replace('agreement', 'agr').replace('induction', 'ind').replace('capital', 'cap')}")
    ax.set_xticklabels(lab, fontsize=6.5, rotation=60, ha="right", rotation_mode="anchor")
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, handlelength=1.2, columnspacing=0.8)
    ax.set_ylim(-4, 112)

    sd, ab = [], []
    for k in cells:
        for t in M["cells"][k]["matched"]["per_trial"]:
            if t["file"] in M["edges"]:
                sd.append(M["edges"][t["file"]]["sd_ratio"]); ab.append(t["above"])
    sd, ab = np.array(sd), np.array(ab, bool)
    ax2.semilogx(sd[~ab], np.full((~ab).sum(), 0) + np.random.default_rng(0).normal(0, .04, (~ab).sum()),
                 "o", ms=3, color="0.6", alpha=.7, label="reverse end within envelope")
    ax2.semilogx(sd[ab], np.full(ab.sum(), 1) + np.random.default_rng(1).normal(0, .04, ab.sum()),
                 "o", ms=4, color="C0", label="reverse end escapes")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["no escape", "escape"], fontsize=7)
    ax2.set_xlabel(r"realised $A\to B$ edge strength", fontsize=7)
    ax2.set_ylim(-0.45, 1.45)
    ax2.legend(fontsize=6.5, frameon=False, loc="center left", numpoints=1,
               handlelength=1.0, handletextpad=0.5)
    ax2.set_title("matched reverse escapes tend to\n"
                  "occur at stronger edges", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


def fig_scales(path):
    """Section 7: per capture, the single-site and joint working-model envelopes and the executed RMS T_C."""
    A = _load("attribution_adequacy.json")
    R = _load("replication_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.4), sharey=True,
                             gridspec_kw={"width_ratios": [len(A["groups"]["published"]["rows"]),
                                                           len(A["groups"]["controls"]["rows"])]})
    panels = ((axes[0], "published", "published S-Inhibition $\\to$ Name Mover pairs", "C0"),
              (axes[1], "controls", "control pairs (no claimed IOI role)", "C0"))
    for ax, grp, ttl, col in panels:
        rows = sorted(A["groups"][grp]["rows"], key=lambda r: r["d3_S"]["env_med"])
        x = np.arange(len(rows))
        ax.semilogy(x, [r["d3_S"]["env_med"] for r in rows], "^", ms=3.5, color=col,
                    label="single-site width")
        ax.semilogy(x, [r["d3_J"]["env_med"] for r in rows], "s", ms=3, mfc="none", color=col,
                    label="joint width")
        ax.semilogy(x, [r["rms_T"] for r in rows], "o", ms=3.2, color="C3", label="executed RMS $T_C$")
        ax.set_title(ttl, fontsize=7.5)
        ax.set_xticks([])
        ax.set_xlabel("captures, sorted by single-site width", fontsize=7)
    if R:
        fl = R["V2"]["published"]["median_floor"]
        axes[0].axhline(fl, color="0.5", lw=0.7, ls=":")
        axes[0].text(0.03, fl * 0.7, "replication floor (scale only)", fontsize=6, color="0.35", va="top",
                     transform=axes[0].get_yaxis_transform())
        axes[0].set_ylim(bottom=fl / 10)
    axes[0].set_ylabel("logits (log scale)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=1.2)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    OUT = os.path.abspath(os.path.join(HERE, "..", "figures"))
    os.makedirs(OUT, exist_ok=True)
    fig_reachable(os.path.join(OUT, "fig_reachable.pdf"))
    fig_regimes(os.path.join(OUT, "fig_regimes.pdf"))
    fig_spectrum(os.path.join(OUT, "fig_spectrum.pdf"))
    fig_matched(os.path.join(OUT, "fig_matched.pdf"))
    fig_scales(os.path.join(OUT, "fig_scales.pdf"))
