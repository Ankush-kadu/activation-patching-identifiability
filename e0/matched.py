"""SUPPORT-MATCHED INTERVENTION FAMILIES  (added 2026-08-27, round-5 audit bug #13).

THE PROBLEM.  The frozen protocol sweeps the single-site family over interpolation coefficients
alpha in linspace(-0.5, 1.5, 7) -- it EXTRAPOLATES 50% past the donor at both ends -- while the
joint family uses RAW donors, i.e. alpha = 1 exactly.  Measured on the frozen captures, the
single-site family spans about 1.9x the joint family's range in the chart.  A degree-3
polynomial that is large only in that extrapolated region then looks "visible to S, blind to J"
for a reason that has nothing to do with any mechanism, and the same-family null cannot detect
it because both halves of S share the alpha support.

THE FIX.  Give the joint family the SAME alpha grid.  Every joint state is
    A <- Aclean + alpha_A (donor_i - Aclean),    B <- Bclean + alpha_B (donor_j - Bclean)
with (alpha_A, alpha_B) taken from the same 7-point grid the single-site family uses, assigned
by a fixed rule over the donor-pair index so the joint family covers the same alpha box.  The
two families then differ ONLY in what the theory says they differ in: single-site patching
reaches a union of two curves (the un-patched coordinate is whatever the network computes),
joint patching reaches a product grid.  That is the phenomenon; the support gap was not.

The frozen alpha = 1 joint family is collected in the same pass as `Zj` so the old and new
protocols can be compared on identical prompts, donors and weights.

NUMERICS.  GPT-2 is captured in float32.  The frozen IOI captures used bfloat16
(adjudicate.py:113), which left the readout with only ~10 distinct values over 448 rows; the
readout resolution of every capture is measured here and stored.
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from levelC2 import Probe, batched, DEV, FULL_ATTN, kind, TEMPLATES
from cells import model_tag, N_AL, ALPHAS
from nonioi import TASKS, SCREEN_STD, N_DON

AL = np.array(ALPHAS)
STEP = 2                      # donor stride for the joint grid -> 16 x 16 = 256 pairs


def edge_chart(A_arm, B_arm, n_train=None, B_alt=None, return_diag=False):
    """Pick the chart that EXPOSES the A -> B edge, using no readout.

    The unembedding chart u = W_U[IO] - W_U[S] is chosen to read OUTPUT relevance; it is not
    chosen to capture what A writes into B's input.  Measured on the frozen captures the realised
    edge in that chart is nearly flat (median whitened slope 0.03), and that single fact explains
    why the recompute and clamped designs nearly coincide, why the theory-named generator reduces
    to the trivial cross generator, and why the fitted interaction coefficient is ~0.

    The edge is a property of the STATES, so the chart can be chosen without ever looking at the
    readout: take the top singular pair of the cross-covariance between the two sites'
    activations along the patch-A arm -- the direction in A's output space and the direction in
    B's output space whose projections co-vary most.  Fitted on a TRAIN half and then frozen.
    """
    n = len(A_arm) if n_train is None else n_train
    Ac = A_arm[:n] - A_arm[:n].mean(0)
    Bc = B_arm[:n] - B_arm[:n].mean(0)
    rank_b, in_range, valid = None, float("nan"), True
    if B_alt is None:                       # covariance-aligned: co-vary most along the A arm
        U, _, Vt = np.linalg.svd(Ac.T @ Bc / max(n, 1), full_matrices=False)
        p, q = U[:, 0], Vt[0]
    else:
        # Proposition 1's hypothesis is gated by sd(b | arm A) / sd(b | arm B), so choose the
        # B-direction that MAXIMISES that ratio: the top generalized eigenvector of the pencil
        # (Cov(B | arm A), Cov(B | arm B)).  Same instrument as the rest of the paper, and it
        # uses only states.  The A-direction then maximises covariance with that coordinate.
        #
        # ROOT-CAUSE GUARD (2026-08-28).  That denominator is built from at most N_DON/2 donor
        # directions in d_model dimensions, so it is rank-deficient BY CONSTRUCTION and the
        # maximiser can land in its null space, where the ridge rather than the data sets the
        # value.  Returning such a direction silently is what produced eleven unusable captures,
        # found only downstream and only by a symptom.  Measure the support here, at the source.
        from scipy.linalg import eigh as _eigh
        Ba, Bb = Bc, B_alt - B_alt.mean(0)
        Ca = Ba.T @ Ba / max(len(Ba), 1)
        Cb = Bb.T @ Bb / max(len(Bb), 1)
        lam = 1e-6 * (np.trace(Ca) + np.trace(Cb)) / (2 * Ca.shape[0])
        I = lam * np.eye(Ca.shape[0])
        _, V = _eigh(Ca + I, Cb + I)
        q = V[:, -1]
        p = Ac.T @ (Ba @ q)
        ev, U = np.linalg.eigh(Cb)
        keep = ev > 1e-10 * max(ev.max(), 1e-300)
        rank_b = int(keep.sum())
        qn = q / np.linalg.norm(q)
        in_range = float(np.linalg.norm(U[:, keep].T @ qn))   # 1.0 = fully supported by the data
        valid = in_range > 0.10
    p, q = p / np.linalg.norm(p), q / np.linalg.norm(q)
    if return_diag:
        return p, q, dict(denominator_rank=rank_b, direction_in_range=in_range, valid=bool(valid))
    return p, q


def alpha_pair(i, j):
    """Deterministic (alpha_A, alpha_B) for donor pair (i, j), covering the same grid as S."""
    return int((i + j) % N_AL), int((3 * i + j) % N_AL)


def resolution(y):
    """How many distinct readout values, and the quantization step relative to sd(Y)."""
    u = np.unique(np.asarray(y, dtype=np.float64))
    step = float(np.min(np.diff(u))) if len(u) > 1 else float("nan")
    return dict(n_distinct=int(len(u)), n=int(len(y)),
                step_over_sd=float(step / max(np.std(y), 1e-300)))


def collect(model, tok, modfn, lA, lB, seed, taskname, names, bs=8, chart_from=None):
    """Single-site (S), alpha-matched joint (Jm) and frozen raw-donor joint (J) families."""
    rng = np.random.default_rng(seed)
    if taskname == "ioi":
        t = TEMPLATES[rng.integers(len(TEMPLATES))]
        a, b = rng.choice(names, 2, replace=False)
        prompt = t.format(A=a, B=b)
        base = tok(prompt, add_special_tokens=False).input_ids
        io = tok(" " + a, add_special_tokens=False).input_ids[0]
        sb = tok(" " + b, add_special_tokens=False).input_ids[0]
        donors = [tok(t.format(A=x, B=y), add_special_tokens=False).input_ids
                  for x, y in [rng.choice(names, 2, replace=False) for _ in range(N_DON)]]
    else:
        got = None
        for _ in range(40):
            got = TASKS[taskname](rng, tok)
            if got is not None:
                break
        if got is None:
            raise RuntimeError("task could not produce single-token targets")
        base, io, sb, donors = got
        prompt = tok.decode(base)
    L = len(base)
    donors = [d for d in donors if len(d) == L]
    if len(donors) < 8:
        raise RuntimeError(f"only {len(donors)} length-matched donors")

    ids = torch.tensor([base], device=DEV)
    p = Probe(model, lA, lB, modfn)
    Ac, Bc, Y0 = batched(model, p, ids, io, sb, bs=1)     # Y0: the CLEAN readout
    Aclean, Bclean = Ac[0], Bc[0]

    dA, dB = [], []
    for i in range(0, len(donors), 8):
        pid = torch.tensor(donors[i:i + 8], device=DEV)
        p.force = {}
        model(pid)
        dA.append(p.cache["A"].float().cpu().numpy().astype(np.float64))
        dB.append(p.cache["B"].float().cpu().numpy().astype(np.float64))
    dA, dB = np.vstack(dA), np.vstack(dB)
    n = len(dA)
    T = lambda X: torch.tensor(X, dtype=next(model.parameters()).dtype, device=DEV)

    # ---- S: single-site, both arms, full alpha grid (donor-major, alpha-minor) ----
    A1, B1, Y1 = batched(model, p, ids, io, sb,
                         fA=T(np.stack([Aclean + x * (v - Aclean) for v in dA for x in AL])), bs=bs)
    A2, B2, Y2 = batched(model, p, ids, io, sb,
                         fB=T(np.stack([Bclean + x * (v - Bclean) for v in dB for x in AL])), bs=bs)

    # ---- C: the CLAMPED single-site family, same alpha grid ----
    # Proposition 1 is about recompute vs clamped, and neither reachable set contains the other
    # when the realised edge has m != 0.  R and C share the patch-B arm and differ only on the
    # patch-A arm: under recompute b is whatever the network computes, under clamping b is held
    # at its clean value.  This is the comparison in which incomparability is PREDICTED, and it
    # is the one the earlier experiments never ran on real states.
    A1c, B1c, Y1c = batched(model, p, ids, io, sb,
                            fA=T(np.stack([Aclean + x * (v - Aclean) for v in dA for x in AL])),
                            fB=T(np.tile(Bclean, (len(dA) * N_AL, 1))), bs=bs)

    # ---- J matched: same donor pairs, alphas from the SAME grid ----
    pairs = [(i, j) for i in range(0, n, STEP) for j in range(0, n, STEP)]
    aidx = np.array([alpha_pair(i // STEP, j // STEP) for i, j in pairs])
    fAm = np.stack([Aclean + AL[ka] * (dA[i] - Aclean) for (i, _), ka in zip(pairs, aidx[:, 0])])
    fBm = np.stack([Bclean + AL[kb] * (dB[j] - Bclean) for (_, j), kb in zip(pairs, aidx[:, 1])])
    A3, B3, Y3 = batched(model, p, ids, io, sb, fA=T(fAm), fB=T(fBm), bs=bs)

    # ---- J frozen: raw donors (alpha = 1), the protocol used in the submitted draft ----
    A4, B4, Y4 = batched(model, p, ids, io, sb,
                         fA=T(np.stack([dA[i] for i, _ in pairs])),
                         fB=T(np.stack([dB[j] for _, j in pairs])), bs=bs)
    p.close()

    W = model.lm_head.weight
    u = (W[io] - W[sb]).float().cpu().numpy().astype(np.float64)
    u /= np.linalg.norm(u)
    # edge-aligned chart, fitted on the first half of the patch-A arm only, then frozen
    pA, qB, chart_diag = edge_chart(A1, B1, n_train=len(A1) // 2,
                                    B_alt=B2[:len(B2) // 2], return_diag=True)
    # CROSS-SEED CONTROL.  The chart is fitted on states, and holding out donors still leaves it
    # fitted on this capture's own prompt and clean activation.  The decisive control is to adopt
    # a chart fitted on a DIFFERENT seed -- different prompt, different donors, same model -- and
    # ask whether the effect survives.  When a sibling capture for this layer pair exists we also
    # project every family onto its chart.  Nothing here touches the readout.
    xchart = None
    if chart_from and os.path.exists(chart_from):
        _d = np.load(chart_from)
        if "pA" in _d.files:
            xchart = (_d["pA"], _d["qB"])
    if not chart_diag["valid"]:
        print(f"    WARNING: edge chart is not supported by the data "
              f"(direction_in_range={chart_diag['direction_in_range']:.3g}, "
              f"denominator rank {chart_diag['denominator_rank']}) -- capture will be screened", flush=True)
    Zs = np.stack([np.concatenate([A1 @ u, A2 @ u]), np.concatenate([B1 @ u, B2 @ u])], 1)
    Ys = np.concatenate([Y1, Y2])
    # clamped family: clamped A-arm + the SAME patch-B arm
    Zc = np.stack([np.concatenate([A1c @ u, A2 @ u]), np.concatenate([B1c @ u, B2 @ u])], 1)
    Yc = np.concatenate([Y1c, Y2])
    # every family, expressed in the edge-aligned chart as well
    Zse = np.stack([np.concatenate([A1 @ pA, A2 @ pA]), np.concatenate([B1 @ qB, B2 @ qB])], 1)
    Zce = np.stack([np.concatenate([A1c @ pA, A2 @ pA]), np.concatenate([B1c @ qB, B2 @ qB])], 1)
    Zjme = np.stack([A3 @ pA, B3 @ qB], 1)
    Zje = np.stack([A4 @ pA, B4 @ qB], 1)
    xproj = {}
    if xchart is not None:
        xp, xq = xchart
        xproj = dict(
            Zse_x=np.stack([np.concatenate([A1 @ xp, A2 @ xp]),
                            np.concatenate([B1 @ xq, B2 @ xq])], 1),
            Zce_x=np.stack([np.concatenate([A1c @ xp, A2 @ xp]),
                            np.concatenate([B1c @ xq, B2 @ xq])], 1),
            Zjme_x=np.stack([A3 @ xp, B3 @ xq], 1))
    arm = np.concatenate([np.zeros(len(A1), int), np.ones(len(A2), int)])
    alph = np.concatenate([np.tile(np.arange(N_AL), len(dA)), np.tile(np.arange(N_AL), len(dB))])
    return dict(Zs=Zs, Ys=Ys, Zc=Zc, Yc=Yc, arm=arm, alpha=alph,
                pA=pA, qB=qB, **xproj,
                Zse=Zse, Zce=Zce, Zjme=Zjme, Zje=Zje,
                Zjm=np.stack([A3 @ u, B3 @ u], 1), Yjm=Y3,
                alphaJA=aidx[:, 0], alphaJB=aidx[:, 1],
                Zj=np.stack([A4 @ u, B4 @ u], 1), Yj=Y4,
                prompt=prompt, clean=float(Y0[0]),
                res_S=resolution(Ys), res_Jm=resolution(Y3), res_J=resolution(Y4),
                res_C=resolution(Yc), chart_valid=chart_diag["valid"],
                chart_in_range=chart_diag["direction_in_range"],
                chart_denom_rank=chart_diag["denominator_rank"] or -1)


def box_overlap(Zs, Zj):
    """Range of the single-site family relative to the joint family, per coordinate."""
    return [float(np.ptp(Zs[:, k]) / max(np.ptp(Zj[:, k]), 1e-300)) for k in (0, 1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ioi"] + list(TASKS))
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpt2", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="capture in float32 (default for gpt2)")
    A = ap.parse_args()

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(A.model)
    dtype = torch.float32 if (A.fp32 or A.gpt2) else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=dtype, device_map=DEV).eval()
    if dtype is torch.float32:
        # a bf16 checkpoint's own config can win over the `dtype` argument, leaving the readout
        # quantized to ~1/16 (observed on Qwen3.5-9B-Base); cast explicitly and verify.
        model = model.float()
    got = next(model.parameters()).dtype
    assert got is dtype, f"requested {dtype} but model loaded as {got}"
    # Architecture dispatch.  Qwen3.5-9B-Base is a HYBRID (gated linear attention on most layers,
    # full attention on FULL_ATTN); Qwen3-8B is dense, so every layer carries self_attn.  Probing
    # the wrong module would silently read a different site, so pick by what the model has.
    if A.gpt2:
        modfn = lambda m, l: m.transformer.h[l].attn
    elif all(hasattr(m_l, "self_attn") for m_l in model.model.layers):
        modfn = lambda m, l: m.model.layers[l].self_attn
    else:
        modfn = lambda m, l: (m.model.layers[l].self_attn if l in FULL_ATTN
                              else m.model.layers[l].linear_attn)
    names = [x for x in ["Mary", "John", "Tom", "Sarah", "Alice", "Bob", "Emma", "David",
                         "Laura", "Peter", "Anna", "James", "Linda", "Paul", "Susan", "Mark"]
             if len(tok(" " + x, add_special_tokens=False).input_ids) == 1]
    MTAG = model_tag(A.model)
    t0 = time.time()
    rows, screened = [], 0
    print(f"=== {A.task}  {A.model}  dtype={dtype}  MATCHED-SUPPORT PROTOCOL ===")
    print(f"{'pair':>8} {'seed':>5} {'sd(Yjm)':>9} {'S/Jm box':>18} {'S/J box':>18} "
          f"{'distinct Y (S)':>15}")
    for pr in A.pairs.split(","):
        lA, lB = (int(x) for x in pr.split("-"))
        for sd in (int(s) for s in A.seeds.split(",")):
            # the first seed of this pair supplies the cross-seed chart for the others
            sib = f"{A.task}_matched_{MTAG}_{lA}_{lB}_{A.seeds.split(',')[0]}.npz"
            try:
                d = collect(model, tok, modfn, lA, lB, sd, A.task, names,
                            chart_from=(None if str(sd) == A.seeds.split(",")[0] else sib))
            except Exception as ex:
                print(f"{f'{lA}-{lB}':>8} {sd:>5}  ERROR {type(ex).__name__}: {str(ex)[:60]}")
                continue
            sdy = float(np.std(d["Yjm"]))
            om, oj = box_overlap(d["Zs"], d["Zjm"]), box_overlap(d["Zs"], d["Zj"])
            print(f"{f'{lA}-{lB}':>8} {sd:>5} {sdy:>9.3f} "
                  f"[{om[0]:>6.2f},{om[1]:>6.2f}]  [{oj[0]:>6.2f},{oj[1]:>6.2f}]  "
                  f"{d['res_S']['n_distinct']:>10}/{d['res_S']['n']}"
                  f"{'   SCREENED' if sdy < SCREEN_STD else ''}")
            if sdy < SCREEN_STD:
                screened += 1
                continue
            out = f"{A.task}_matched_{MTAG}_{lA}_{lB}_{sd}.npz"
            np.savez(out, **{k: v for k, v in d.items() if k not in ("prompt",)})
            rows.append(dict(pair=[lA, lB], seed=sd, std_Yjm=sdy, file=out,
                             box_S_over_Jm=om, box_S_over_J=oj,
                             res_S=d["res_S"], res_Jm=d["res_Jm"], res_J=d["res_J"],
                             prompt=d["prompt"], clean=d["clean"]))
    summ = dict(task=A.task, model=A.model, dtype=str(dtype), n_kept=len(rows),
                n_screened_out=screened, rows=rows)
    fn = f"matched_{A.task}_{MTAG}.json"
    json.dump(summ, open(fn, "w"), indent=1)
    print(f"\nkept {len(rows)}, screened {screened}  ->  {fn}  [{time.time()-t0:.0f}s]")
