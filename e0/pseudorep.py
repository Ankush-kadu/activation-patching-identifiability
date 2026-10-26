"""PSEUDO-REPLICATION IN THE THREE-HEAD-GROUP COMPARISON  (added 2026-09-23, review response).

What is being corrected.  The IOI attribution analysis compares the realised edge of the published S-Inhibition ->
Name Mover heads with early "control" heads -> the same Name Movers, and reports

    QCEdgeVsCtrl  = median(sd_ratio | IOI configs) / median(sd_ratio | control configs)   = 43.3
    QCEdgeVsCtrlP = one-sided Mann-Whitney p, IOI configs > control configs             = 2.35e-11
    GrpIOIRejS / GrpCtrlRejS = single-site Wald rejections at 0.05                       = 26/31, 0/28

(the ledger, "qc.ctrl.*", reading ioi_query_chart{,_control}.json; the counts come from
circuits_summary.py).  Every one of those treats a (source head, Name Mover, seed) configuration as
an independent unit.  It is not: the design is 4 source heads x 3 Name Movers x 3 seeds per group,
the SAME 3 Name Movers serve both groups, and a seed fixes the prompt and donor set for every head
pair (collect() draws them from default_rng(seed) before touching the heads), so configurations
share a source head, a Name Mover and a prompt.  The group label varies only BETWEEN SOURCE HEADS,
so there are 4 + 4 units of replication for the group contrast, not 31 + 28.

What this script does, from the saved captures (qchart_/qctrl_/qind_*.npz):
  0. re-derives every configuration's realised edge and Wald p from the captures with the
     project's own code (identifiability.analyse) and checks them against the JSON rows the paper
     reads, then reproduces the pseudo-replicated numbers;
  1. HEAD-LEVEL test: each source head -> one number (mean over its Name Movers of the mean over
     seeds), then exact tests on 4 vs 4 heads;
  2. HEAD-BLOCK PERMUTATION: relabel whole source heads (all their Name Mover x seed rows move
     together; every Name Mover x seed cell keeps its composition), recomputing (a) the paper's own
     configuration-level Mann-Whitney U and (b) a Name-Mover-stratified mean difference.  Exact over
     all C(8,4)=70 relabelings; valid under any dependence induced by shared Name Movers or prompts;
  3. TWO-WAY CLUSTER-ROBUST OLS (Cameron-Gelbach-Miller) clustering on source head AND Name Mover,
     with Name Mover and seed fixed effects -- reported with its small-cluster caveat;
  4. whether the ORDERING IOI > induction > control survives at the head level (exact
     Jonckheere-Terpstra over 3150 assignments, and the pairwise tests the ordering needs);
  5. the head-level analogue of the pooled Spearman(edge, information gap) of circuits_summary.py.

No held-out claim is made here and nothing is fitted across configurations: each configuration's
Wald p is the project's in-configuration statistic (whitening fitted on that configuration's own
states, as in ioi_query_chart.py), reused unchanged.
"""
import glob, itertools, json, os
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact, ttest_ind, spearmanr, norm
from scipy.stats import t as tdist

from circuits_summary import GROUPS                                  # group -> JSON the paper reads
from ioi_query_chart import S_INHIB, NAME_MOVE, PREV_TOK, INDUCTION  # head lists, weights-only
from identifiability import analyse                                  # the per-configuration test

os.chdir(os.path.dirname(os.path.abspath(__file__)))
CTRL_A = [(0, 5), (1, 2), (2, 9), (3, 4)]   # copied verbatim: defined inside __main__ of
                                            # ioi_query_chart.py, so it cannot be imported
SRC = {"ioi": S_INHIB, "induction": PREV_TOK, "control": CTRL_A}
TGT = {"ioi": NAME_MOVE, "induction": INDUCTION, "control": NAME_MOVE}
PREFIX = {"ioi": "qchart", "induction": "qind", "control": "qctrl"}
ALPHA = 0.05
lab = lambda h: f"L{h[0]}H{h[1]}"


def match_head(tok, heads):
    """File names concatenate layer and head digits ('810' = L8H10, '100' = L10H0); resolve
    against the known head list so the parse is unambiguous."""
    hit = [h for h in heads if f"{h[0]}{h[1]}" == tok]
    assert len(hit) == 1, (tok, heads)
    return hit[0]


# ---------------------------------------------------------------- 0. units, from the captures
def load_units():
    units, checks = [], dict(n=0, max_rel_sd=0.0, max_abs_log10p_S=0.0, max_abs_log10p_J=0.0,
                             max_rel_info=0.0, json_label_mismatch=[])
    for g in ("ioi", "induction", "control"):
        J = json.load(open(GROUPS[g]))["rows"]
        used = set()
        for f in sorted(glob.glob(f"{PREFIX[g]}_*.npz")):
            _, s_tok, t_tok, seed = os.path.basename(f)[:-4].split("_")
            si, nm, seed = match_head(s_tok, SRC[g]), match_head(t_tok, TGT[g]), int(seed)
            d = np.load(f)
            n = len(d["Zs"]) // 2
            # the realised edge exactly as ioi_query_chart.py defines it (inline there, one line)
            sd = float(np.std(d["Zs"][:n, 1]) / max(np.std(d["Zs"][n:, 1]), 1e-300))
            r = analyse(d["Zs"], d["Ys"], d["Zjm"], d["Yjm"])
            # the JSON row the paper reads: same seed, same realised edge
            k = [i for i, row in enumerate(J) if row["seed"] == seed and i not in used
                 and abs(row["sd_ratio"] - sd) <= 1e-9 * max(sd, 1e-12)]
            assert len(k) == 1, (f, k)
            used.add(k[0]); row = J[k[0]]
            if tuple(row["si"]) != si or tuple(row["nm"]) != nm:
                checks["json_label_mismatch"].append(
                    dict(file=os.path.basename(f), json_si=row["si"], json_nm=row["nm"],
                         capture_si=list(si), capture_nm=list(nm)))
            checks["n"] += 1
            checks["max_rel_sd"] = max(checks["max_rel_sd"], abs(row["sd_ratio"] - sd) / sd)
            for key, ck in (("wald_p_S", "max_abs_log10p_S"), ("wald_p_J", "max_abs_log10p_J")):
                checks[ck] = max(checks[ck], abs(np.log10(max(row[key], 1e-300))
                                                 - np.log10(max(r[key], 1e-300))))
            checks["max_rel_info"] = max(checks["max_rel_info"],
                                         abs(row["info_ratio"] - r["info_ratio"]) / r["info_ratio"])
            units.append(dict(group=g, src=lab(si), tgt=lab(nm), seed=seed, sd_ratio=sd,
                              wald_p_S=float(r["wald_p_S"]), wald_p_J=float(r["wald_p_J"]),
                              info_ratio=float(r["info_ratio"]),
                              rej_S=int(r["wald_p_S"] < ALPHA), rej_J=int(r["wald_p_J"] < ALPHA)))
        assert len(used) == len(J), (g, len(used), len(J))
    return units, checks


def field(u, k):
    return np.log10(u["sd_ratio"]) if k == "log_edge" else (
        np.log10(u["info_ratio"]) if k == "log_info" else float(u[k]))


def head_values(units, k):
    """Source head -> mean over its target heads of (mean over seeds): each target head gets equal
    weight whatever the screen left, and a head with 9 configurations counts once."""
    out = {}
    for h in sorted({u["src"] for u in units}):
        cells = {}
        for u in units:
            if u["src"] == h:
                cells.setdefault(u["tgt"], []).append(field(u, k))
        out[h] = float(np.mean([np.mean(v) for v in cells.values()]))
    return out


# ---------------------------------------------------------------- head-block permutation machinery
def relabelings(heads, n_first):
    """Every way of calling n_first of the heads 'group 1' -- exact, not sampled."""
    return [set(c) for c in itertools.combinations(heads, n_first)]


def perm_p(stat, heads, obs_set, n_first):
    """One- and two-sided exact p of stat(set of heads labelled group 1) over all relabelings."""
    L = relabelings(heads, n_first)
    T = np.array([stat(s) for s in L]); t0 = stat(obs_set)
    eps = 1e-12 * max(1.0, abs(t0))
    return dict(stat=float(t0), n_relabelings=len(L),
                p_greater=float(np.mean(T >= t0 - eps)),
                p_two_sided=float(np.mean(np.abs(T - np.mean(T)) >= abs(t0 - np.mean(T)) - eps)),
                p_floor=1.0 / len(L))


def U_stat(x, y):
    """Mann-Whitney U of x over y (ties 1/2) -- the statistic behind QCEdgeVsCtrlP."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(np.sum(x > y) + 0.5 * np.sum(x == y))


def head_level_block(units, k, g1, g2):
    """Head-level summary + exact tests for group g1 > group g2 on per-configuration field k."""
    U = [u for u in units if u["group"] in (g1, g2)]
    hv = head_values(U, k)
    h1 = sorted({u["src"] for u in U if u["group"] == g1})
    h2 = sorted({u["src"] for u in U if u["group"] == g2})
    heads = h1 + h2
    a, b = np.array([hv[h] for h in h1]), np.array([hv[h] for h in h2])
    mean_diff = lambda s: np.mean([hv[h] for h in s]) - np.mean([hv[h] for h in heads if h not in s])
    # (b) Name-Mover-stratified difference over configurations: within each target head compare the
    # configurations whose source head carries label 1 with the rest, then average over targets
    tg = sorted({u["tgt"] for u in U})
    def strat(s):
        dd = []
        for t in tg:
            x = [field(u, k) for u in U if u["tgt"] == t and u["src"] in s]
            y = [field(u, k) for u in U if u["tgt"] == t and u["src"] not in s]
            if x and y:
                dd.append(np.mean(x) - np.mean(y))
        return float(np.mean(dd))
    def U_conf(s):
        return U_stat([field(u, k) for u in U if u["src"] in s],
                      [field(u, k) for u in U if u["src"] not in s])
    res = dict(n_heads=[len(h1), len(h2)],
               n_configs=[sum(u["group"] == g1 for u in U), sum(u["group"] == g2 for u in U)],
               head_values={g1: {h: hv[h] for h in h1}, g2: {h: hv[h] for h in h2}},
               head_mean_diff=float(a.mean() - b.mean()),
               complete_separation=bool(a.min() > b.max()),
               min_g1_head=float(a.min()), max_g2_head=float(b.max()),
               perm_head_mean_diff=perm_p(mean_diff, heads, set(h1), len(h1)),
               perm_conf_U=perm_p(U_conf, heads, set(h1), len(h1)))
    # stratifying by target head only means something when both groups hit the same targets
    if {u["tgt"] for u in U if u["group"] == g1} == {u["tgt"] for u in U if u["group"] == g2}:
        res["perm_nm_stratified"] = perm_p(strat, heads, set(h1), len(h1))
    if len(set(np.r_[a, b])) == len(a) + len(b):
        res["mwu_exact_heads_p_greater"] = float(mannwhitneyu(a, b, alternative="greater",
                                                              method="exact").pvalue)
    if a.std() + b.std() > 0:
        tt = ttest_ind(a, b, equal_var=False, alternative="greater")
        res["welch_t_heads"] = dict(t=float(tt.statistic), df=float(tt.df), p_greater=float(tt.pvalue))
    return res


# ---------------------------------------------------------------- two-way cluster-robust OLS
def cgm_ols(units, k, g1="ioi", g0="control"):
    """y = b0 + b1*[source head is g1] + target-head FE + seed FE; Cameron-Gelbach-Miller two-way
    clustered variance V_head + V_tgt - V_head&tgt, each with factor G/(G-1)*(N-1)/(N-K)."""
    U = [u for u in units if u["group"] in (g1, g0)]
    y = np.array([field(u, k) for u in U])
    tg, sds = sorted({u["tgt"] for u in U}), sorted({u["seed"] for u in U})
    X = np.column_stack([np.ones(len(U)), [u["group"] == g1 for u in U]]
                        + [[u["tgt"] == t for u in U] for t in tg[1:]]
                        + [[u["seed"] == s for u in U] for s in sds[1:]]).astype(float)
    N, K = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    def V(cl):
        cl = np.asarray(cl); G = len(set(cl.tolist()))
        M = np.zeros((K, K))
        for c in set(cl.tolist()):
            s = X[cl == c].T @ e[cl == c]; M += np.outer(s, s)
        return G / (G - 1) * (N - 1) / (N - K) * XtXi @ M @ XtXi, G
    Vh, Gh = V([u["src"] for u in U])
    Vt, Gt = V([u["tgt"] for u in U])
    Vht, Ght = V([u["src"] + u["tgt"] for u in U])
    V2 = Vh + Vt - Vht
    out = dict(N=N, K=K, coef=float(b[1]), G_head=Gh, G_tgt=Gt, G_head_x_tgt=Ght)
    se1 = float(np.sqrt(Vh[1, 1]))
    out["oneway_head"] = dict(se=se1, t=float(b[1] / se1), df=Gh - 1,
                              p_greater=float(tdist.sf(b[1] / se1, Gh - 1)))
    v2 = float(V2[1, 1])
    if v2 > 0:
        se2 = np.sqrt(v2); t2 = float(b[1] / se2); df2 = min(Gh, Gt) - 1
        out["twoway_head_tgt"] = dict(se=float(se2), t=t2, df=df2,
                                      p_greater_t=float(tdist.sf(t2, df2)),
                                      p_greater_normal=float(norm.sf(t2)))
    else:
        out["twoway_head_tgt"] = dict(variance=v2, note="two-way variance not positive; undefined")
    return out


# ---------------------------------------------------------------- ordering, 3 groups, head level
def jt_exact(hv, groups_by_head, order):
    """Exact Jonckheere-Terpstra over every assignment of the observed head values to groups of the
    observed sizes; alternative: values increase along `order` (low -> high)."""
    heads = list(hv)
    sizes = [sum(groups_by_head[h] == g for h in heads) for g in order]
    def jt(assign):
        s = 0.0
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                s += U_stat([hv[h] for h in heads if assign[h] == order[j]],
                            [hv[h] for h in heads if assign[h] == order[i]])
        return s
    t0 = jt(groups_by_head)
    T = []
    for top in itertools.combinations(heads, sizes[2]):
        rest = [h for h in heads if h not in top]
        for mid in itertools.combinations(rest, sizes[1]):
            a = {h: order[0] for h in heads}
            a.update({h: order[1] for h in mid}); a.update({h: order[2] for h in top})
            T.append(jt(a))
    T = np.array(T)
    return dict(stat=float(t0), max_possible=float(sum(sizes[i] * sizes[j] for i in range(3)
                                                        for j in range(i + 1, 3))),
                n_assignments=len(T), p_greater=float(np.mean(T >= t0 - 1e-9)), p_floor=1 / len(T))


def spearman_exact(x, y):
    """Exact permutation p of Spearman rho over all n! pairings (n = 10 -> 3,628,800)."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(np.int16)
    ry = np.argsort(np.argsort(y)).astype(np.int16)
    rho0 = 1 - 6 * float(np.sum((rx - ry) ** 2)) / (n * (n * n - 1))
    P = np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(n))),
                    dtype=np.int8).reshape(-1, n)
    d2 = ((rx[None, :] - ry[P]) ** 2).sum(1)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    return dict(rho=rho0, n=n, n_permutations=len(rho),
                p_less=float(np.mean(rho <= rho0 + 1e-12)),
                p_two_sided=float(np.mean(np.abs(rho) >= abs(rho0) - 1e-12)))


if __name__ == "__main__":
    units, checks = load_units()
    by = lambda g: [u for u in units if u["group"] == g]
    I, C, D = by("ioi"), by("control"), by("induction")
    print(f"[PSEUDOREP] captures re-analysed: {checks['n']} "
          f"(ioi {len(I)}, induction {len(D)}, control {len(C)})")
    print(f"[PSEUDOREP] vs the JSON rows the paper reads: max rel diff edge {checks['max_rel_sd']:.1e}, "
          f"max |dlog10 p_S| {checks['max_abs_log10p_S']:.1e}, max |dlog10 p_J| "
          f"{checks['max_abs_log10p_J']:.1e}, max rel diff info ratio {checks['max_rel_info']:.1e}")
    print(f"[PSEUDOREP] JSON head-label mismatches: {len(checks['json_label_mismatch'])}"
          + (f" (e.g. {checks['json_label_mismatch'][0]})" if checks["json_label_mismatch"] else ""))

    # ---- the pseudo-replicated numbers, reproduced exactly as claims.py / circuits_summary.py do
    eI, eC = [u["sd_ratio"] for u in I], [u["sd_ratio"] for u in C]
    pr = dict(unit="configuration (source head x target head x seed), treated as independent",
              n_ioi=len(I), n_control=len(C),
              edge_median_ioi=float(np.median(eI)), edge_median_control=float(np.median(eC)),
              ratio_of_medians=float(np.median(eI) / np.median(eC)),
              U=U_stat(eI, eC), U_max=len(I) * len(C),
              mwu_p_greater=float(mannwhitneyu(eI, eC, alternative="greater").pvalue),
              rej_S=[sum(u["rej_S"] for u in I), sum(u["rej_S"] for u in C)],
              rej_J=[sum(u["rej_J"] for u in I), sum(u["rej_J"] for u in C)],
              rej_S_induction=sum(u["rej_S"] for u in D), rej_J_induction=sum(u["rej_J"] for u in D))
    for k in ("rej_S", "rej_J"):
        a, c = pr[k]
        pr[f"fisher_{k}_p_greater"] = float(fisher_exact([[a, len(I) - a], [c, len(C) - c]],
                                                         alternative="greater").pvalue)
    print("\n== pseudo-replicated reference (unit = configuration)")
    print(f"  edge median IOI {pr['edge_median_ioi']:.4f} / control {pr['edge_median_control']:.5f}"
          f" = {pr['ratio_of_medians']:.1f}x;  U = {pr['U']:.0f}/{pr['U_max']};  "
          f"MWU p(greater) = {pr['mwu_p_greater']:.3g}")
    print(f"  single-site rejections {pr['rej_S'][0]}/{len(I)} vs {pr['rej_S'][1]}/{len(C)}  "
          f"Fisher p = {pr['fisher_rej_S_p_greater']:.3g};  joint {pr['rej_J'][0]}/{len(I)} vs "
          f"{pr['rej_J'][1]}/{len(C)}  Fisher p = {pr['fisher_rej_J_p_greater']:.3g}")

    # ---- 1+2. head as the unit; head-block permutation
    HL = {k: head_level_block(units, k, "ioi", "control") for k in ("log_edge", "rej_S", "rej_J")}
    he = HL["log_edge"]
    gI = {h: 10 ** v for h, v in he["head_values"]["ioi"].items()}
    gC = {h: 10 ** v for h, v in he["head_values"]["control"].items()}
    he["geomean_edge"] = dict(ioi=gI, control=gC)
    he["ratio_of_head_medians"] = float(np.median(list(gI.values())) / np.median(list(gC.values())))
    # sensitivity: a different within-head aggregation (median over the head's configurations)
    medh = lambda G: {h: float(np.median([u["sd_ratio"] for u in G if u["src"] == h]))
                      for h in sorted({u["src"] for u in G})}
    mI, mC = medh(I), medh(C)
    he["sensitivity_median_aggregation"] = dict(
        ioi=mI, control=mC, complete_separation=bool(min(mI.values()) > max(mC.values())),
        mwu_exact_heads_p_greater=float(mannwhitneyu(list(mI.values()), list(mC.values()),
                                                     alternative="greater", method="exact").pvalue),
        ratio_of_head_medians=float(np.median(list(mI.values())) / np.median(list(mC.values()))))
    # per target head: the 3 Name Movers are shared, so they are blocks, not units
    pertgt = {}
    for t in sorted({u["tgt"] for u in I}):
        a = [np.mean([field(u, "log_edge") for u in I if u["tgt"] == t and u["src"] == h])
             for h in sorted({u["src"] for u in I if u["tgt"] == t})]
        c = [np.mean([field(u, "log_edge") for u in C if u["tgt"] == t and u["src"] == h])
             for h in sorted({u["src"] for u in C if u["tgt"] == t})]
        pertgt[t] = dict(log10_edge_diff=float(np.mean(a) - np.mean(c)), sep=bool(min(a) > max(c)))
    he["per_target_head"] = pertgt

    print("\n== IOI vs control, SOURCE HEAD as the unit (4 vs 4)")
    print(f"  {'head':<8}{'group':<9}{'geo-mean edge':>14}{'rej_S rate':>12}{'rej_J rate':>12}")
    for g, hs in (("ioi", gI), ("control", gC)):
        for h, v in hs.items():
            print(f"  {h:<8}{g:<9}{v:>14.5f}{HL['rej_S']['head_values'][g][h]:>12.3f}"
                  f"{HL['rej_J']['head_values'][g][h]:>12.3f}")
    for k, name in (("log_edge", "log10 edge"), ("rej_S", "single-site rejection rate"),
                    ("rej_J", "joint rejection rate")):
        r = HL[k]
        print(f"  {name:<27} diff {r['head_mean_diff']:+.3f}  complete separation "
              f"{r['complete_separation']}  exact head-perm p(greater) "
              f"{r['perm_head_mean_diff']['p_greater']:.4f} (two-sided "
              f"{r['perm_head_mean_diff']['p_two_sided']:.4f}; floor 1/70 = {1/70:.4f})"
              + (f"  exact MWU(heads) p {r['mwu_exact_heads_p_greater']:.4f}"
                 if "mwu_exact_heads_p_greater" in r else "")
              + (f"  Welch t {r['welch_t_heads']['t']:.2f} (df {r['welch_t_heads']['df']:.1f}) "
                 f"p {r['welch_t_heads']['p_greater']:.2g}" if "welch_t_heads" in r else ""))
    print(f"  ratio of head-level medians of geometric-mean edge: {he['ratio_of_head_medians']:.1f}x "
          f"(configuration-level ratio of medians {pr['ratio_of_medians']:.1f}x)")
    s = he["sensitivity_median_aggregation"]
    print(f"  sensitivity (head = median of its configurations): separation "
          f"{s['complete_separation']}, exact MWU p {s['mwu_exact_heads_p_greater']:.4f}, "
          f"ratio {s['ratio_of_head_medians']:.1f}x")
    print("  per shared target head (block), mean log10 edge IOI - control: "
          + ", ".join(f"{t} {v['log10_edge_diff']:+.2f}{' sep' if v['sep'] else ''}"
                      for t, v in pertgt.items()))
    print("\n== head-block permutation (whole source heads relabelled; target x seed cells fixed)")
    for k in ("log_edge", "rej_S", "rej_J"):
        r = HL[k]
        print(f"  {k:<9} configuration-level U {r['perm_conf_U']['stat']:.1f}: p(greater) "
              f"{r['perm_conf_U']['p_greater']:.4f};  target-stratified diff "
              f"{r['perm_nm_stratified']['stat']:+.3f}: p(greater) "
              f"{r['perm_nm_stratified']['p_greater']:.4f}  [{r['perm_conf_U']['n_relabelings']} relabelings]")

    # ---- 3. two-way cluster-robust
    CR = {k: cgm_ols(units, k) for k in ("log_edge", "rej_S", "rej_J")}
    print("\n== two-way cluster-robust OLS (clusters: source head, target head; FE: target head, seed)")
    for k, r in CR.items():
        tw = r["twoway_head_tgt"]
        print(f"  {k:<9} coef {r['coef']:+.3f}  one-way(head, G={r['G_head']}) se {r['oneway_head']['se']:.3f}"
              f" t {r['oneway_head']['t']:.2f} p_t({r['oneway_head']['df']}) {r['oneway_head']['p_greater']:.2g}"
              + (f"  two-way se {tw['se']:.3f} t {tw['t']:.2f} p_t({tw['df']}) {tw['p_greater_t']:.2g}"
                 f" p_normal {tw['p_greater_normal']:.2g}" if "se" in tw else f"  two-way: {tw['note']}"))

    # ---- 4. ordering IOI > induction > control, head level
    ORD = {}
    for k in ("log_edge", "rej_S"):
        hv = head_values(units, k)
        gb = {u["src"]: u["group"] for u in units}
        med = {g: float(np.median([hv[h] for h in hv if gb[h] == g]))
               for g in ("ioi", "induction", "control")}
        grp = lambda g: [hv[h] for h in hv if gb[h] == g]
        ORD[k] = dict(head_values={h: dict(group=gb[h], value=hv[h]) for h in hv},
                      median_of_heads=med,
                      medians_ordered=bool(med["ioi"] > med["induction"] > med["control"]),
                      all_heads_ordered=bool(min(grp("ioi")) > max(grp("induction"))
                                             and min(grp("induction")) > max(grp("control"))),
                      jt=jt_exact(hv, gb, ["control", "induction", "ioi"]),
                      ioi_vs_induction=head_level_block(units, k, "ioi", "induction"),
                      induction_vs_control=head_level_block(units, k, "induction", "control"))
        for g1, g2 in (("ioi", "induction"), ("induction", "control")):
            x = [field(u, k) for u in by(g1)]; y = [field(u, k) for u in by(g2)]
            ORD[k][f"pseudorep_mwu_{g1}_vs_{g2}_p_greater"] = float(
                mannwhitneyu(x, y, alternative="greater").pvalue)
    print("\n== ordering IOI > induction > control, SOURCE HEAD as the unit (4, 2, 4 heads)")
    for k, r in ORD.items():
        m = r["median_of_heads"]
        a, b = r["ioi_vs_induction"], r["induction_vs_control"]
        print(f"  {k:<9} head medians ioi {m['ioi']:+.3f} ind {m['induction']:+.3f} ctrl {m['control']:+.3f}"
              f"  ordered {r['medians_ordered']} (every head {r['all_heads_ordered']});  exact JT p "
              f"{r['jt']['p_greater']:.4f} [floor {r['jt']['p_floor']:.1e}]")
        print(f"            ioi>ind head-perm p {a['perm_head_mean_diff']['p_greater']:.4f}, "
              f"ind>ctrl head-perm p {b['perm_head_mean_diff']['p_greater']:.4f} [floor 1/15 = {1/15:.4f}]"
              f";  pseudo-replicated MWU p {r['pseudorep_mwu_ioi_vs_induction_p_greater']:.2g}, "
              f"{r['pseudorep_mwu_induction_vs_control_p_greater']:.2g}")

    # ---- 5. head-level Spearman(edge, information gap) -- circuits_summary pooled 77 configs
    hE, hG = head_values(units, "log_edge"), head_values(units, "log_info")
    hs = sorted(hE)
    SP = spearman_exact([hE[h] for h in hs], [hG[h] for h in hs])
    rho_c, p_c = spearmanr([u["sd_ratio"] for u in units], [u["info_ratio"] for u in units])
    SP.update(pseudorep_rho=float(rho_c), pseudorep_p=float(p_c), pseudorep_n=len(units))
    within = {}
    for h in hs:
        G = [u for u in units if u["src"] == h]
        within[h] = float(spearmanr([u["sd_ratio"] for u in G], [u["info_ratio"] for u in G])[0])
    SP["within_head_rho"] = within
    SP["within_head_negative"] = f"{sum(v < 0 for v in within.values())}/{len(within)}"
    print(f"\n== Spearman(edge, information gap): pooled configurations rho {rho_c:+.3f} p {p_c:.2g} "
          f"(n={len(units)});  head level rho {SP['rho']:+.3f} exact p(less) {SP['p_less']:.2g} "
          f"(n={SP['n']} heads, {SP['n_permutations']} permutations);  within-head rho < 0 in "
          f"{SP['within_head_negative']} heads (min {min(within.values()):+.2f}, max {max(within.values()):+.2f})")

    # ---- what 4 vs 4 heads can and cannot say, computed rather than asserted
    layers = lambda G: sorted({int(u["src"][1:].split("H")[0]) for u in G})
    reading = dict(
        edge_direction_survives=bool(he["complete_separation"]
                                     and he["perm_head_mean_diff"]["p_greater"] <= ALPHA),
        edge_head_p_is_floor=bool(abs(he["perm_head_mean_diff"]["p_greater"] - 1 / 70) < 1e-12),
        rejS_direction_survives=bool(HL["rej_S"]["complete_separation"]
                                     and HL["rej_S"]["perm_head_mean_diff"]["p_greater"] <= ALPHA),
        rejJ_difference_detected=bool(HL["rej_J"]["perm_head_mean_diff"]["p_greater"] <= ALPHA),
        edge_ordering_medians=ORD["log_edge"]["medians_ordered"],
        edge_ordering_every_head=ORD["log_edge"]["all_heads_ordered"],
        induction_position_testable_at_alpha=bool(1 / 15 <= ALPHA),
        smallest_p_any_ioi_vs_control_head_test=1 / 70,
        smallest_p_any_pairwise_test_with_2_induction_heads=1 / 15,
        source_layers=dict(ioi=layers(I), induction=layers(D), control=layers(C)),
        target_layers=dict(ioi_control=sorted({int(u["tgt"][1:].split("H")[0]) for u in I}),
                           induction=sorted({int(u["tgt"][1:].split("H")[0]) for u in D})))
    print(f"\n== reading: edge direction survives at the head level {reading['edge_direction_survives']}"
          f" (p at the 1/70 floor: {reading['edge_head_p_is_floor']}); single-site rejection contrast "
          f"survives {reading['rejS_direction_survives']}; joint contrast detected "
          f"{reading['rejJ_difference_detected']}; induction's middle place testable at {ALPHA}: "
          f"{reading['induction_position_testable_at_alpha']}; source layers "
          f"{reading['source_layers']} (group is confounded with depth)")

    heads_rows = []
    for h in hs:
        G = [u for u in units if u["src"] == h]
        heads_rows.append(dict(src=h, group=G[0]["group"], n_configs=len(G),
                               n_targets=len({u["tgt"] for u in G}),
                               geomean_edge=10 ** hE[h], log10_info_gap=hG[h],
                               rej_S_rate=head_values(G, "rej_S")[h],
                               rej_J_rate=head_values(G, "rej_J")[h]))
    out = dict(
        estimands=dict(
            edge="realised edge sd(b | arm S) / sd(b | arm N) per configuration, as in ioi_query_chart.py",
            head_value="per source head: mean over its target heads of the mean over seeds, of log10(edge)"
                       " (reported back as a geometric mean) or of the 0/1 Wald rejection at 0.05",
            head_mean_diff="difference of group means of head values (IOI minus control)",
            perm="exact over all relabelings of whole source heads; target-head x seed cells fixed",
            cgm="OLS coef on IOI indicator with target-head and seed FE; Cameron-Gelbach-Miller"
                " two-way (source head, target head) cluster variance, df = min(G)-1"),
        design=dict(ioi="4 S-Inhibition heads x 3 Name Movers x 3 seeds, 31 kept by the screen",
                    control="4 early heads x the same 3 Name Movers x 3 seeds, 28 kept",
                    induction="2 previous-token heads x 3 induction heads x 3 seeds, 18 kept;"
                              " different targets and different (random-token) prompts",
                    shared="a seed fixes the prompt and donors for every head pair of a group"),
        verification=checks, pseudoreplicated=pr, head_level=HL, cluster_robust=CR,
        ordering=ORD, spearman=SP, reading=reading, heads=heads_rows,
        configurations=[{k: (float(f"{v:.7g}") if isinstance(v, float) else v) for k, v in u.items()}
                        for u in units])
    json.dump(out, open("pseudorep.json", "w"), indent=1, default=float)
    print(f"\nsaved -> pseudorep.json ({os.path.getsize('pseudorep.json') // 1024} KB)")
