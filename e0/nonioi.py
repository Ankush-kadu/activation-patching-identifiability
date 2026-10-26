"""GENERALIZATION CHECK: does reciprocal family blindness appear OUTSIDE IOI?

MOTIVATION (external review, 2026-08-19).  Every strong natural-model result so far is on
IOI -- a single, heavily-studied task.  A reviewer can say "you discovered a geometry
peculiar to one synthetic benchmark."  The exact theory is task-free and there are two
models, so this is not existential, but one substantially different behavior materially
improves the paper.

===================== PRE-REGISTRATION (written BEFORE running) =====================
PROTOCOL: FROZEN.  Identical to adjudicate.collect + subspace_test.analyse.  No new tuning,
no method changes, no new hyperparameters.  Same donor count (32), same alpha sweep
(7 points on [-0.5, 1.5]), same degree-3 polynomial class, same 50/50 train/test split,
same generalized eigenproblem G_S v = lam G_J v with delta=1e-10, same >10x criterion.

SCREEN: the pre-specified causal-influence screen std(Y_joint) >= 0.15 (fixed earlier for
Qwen) is applied UNCHANGED.  Trials failing it are reported as screened out, never dropped
silently -- a layer pair that carries no task signal produces noise, not evidence.

THE ONE QUESTION: does the reciprocal result (BOTH an S-blind/J-visible AND a J-blind/
S-visible direction at >10x on held-out data) appear on non-IOI behaviors?

PREDICTION: yes, because the mechanism is a property of the INTERVENTION FAMILY (single-site
sweeps pin the un-patched coordinate at its recomputed value, collapsing cross terms), not of
the task.  The theory predicts blindness wherever the reachable set is a union of slices.

PRE-COMMITTED READINGS:
  BOTH >10x in a MAJORITY of screened trials on >=2 non-IOI tasks  -> claim generalizes;
      state "IOI and three further behaviors".
  BOTH >10x on some tasks but not others                           -> report per task, and
      scope the empirical claim to the tasks where it holds. Do NOT aggregate to hide it.
  BOTH >10x in a MINORITY everywhere                               -> the empirical claim is
      IOI-specific. Say so in the abstract and keep the exact theory as the general result.
Whatever comes out is reported. No task will be added or dropped after seeing results.
=====================================================================================

TASKS -- chosen to be mechanistically unlike IOI, not merely different prompts:
  induction  : random repeated token sequence. NO semantic content, no names, no template.
               Tests the copying/induction mechanism (Elhage et al.).
  greater    : "The war lasted from the year 17XX to the year 17__". Numeric comparison
               (Hanna et al. 2023), a different circuit with strong MLP involvement.
  agreement  : "The keys to the cabinet ___" -> " are" vs " is". Syntactic number agreement
               across an attractor; no entity tracking at all.
  capital    : "The capital of France is" -> " Paris" vs a distractor capital. Factual
               retrieval, known to be MLP-mediated rather than attention-mediated.
"""
import argparse, json, os, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from levelC2 import Probe, batched, DEV, FULL_ATTN, kind
from subspace_test import analyse
from cells import model_tag, raw_name

SCREEN_STD = 0.15          # pre-specified, unchanged from the Qwen protocol
N_DON, N_AL = 32, 7        # frozen


# --------------------------------------------------------------------- tasks ---
def t_induction(rng, tok, seqlen=12):
    lo, hi = 1000, 20000
    def one():
        ids = rng.integers(lo, hi, size=seqlen).tolist()
        return ids + ids[:-1]                      # second copy, stops before the answer
    base = one()
    ids = base[:]
    correct = base[seqlen - 1]                     # the token that followed the cue before
    wrong = base[int(rng.integers(0, seqlen - 2))]
    return ids, correct, wrong, [one() for _ in range(N_DON)]


def t_greater(rng, tok, _=None):
    nouns = ["war", "siege", "famine", "drought", "voyage", "revolt", "plague", "feud"]
    def one():
        n = nouns[int(rng.integers(len(nouns)))]
        yy = int(rng.integers(20, 60))
        return f"The {n} lasted from the year 17{yy:02d} to the year 17", yy
    txt, yy = one()
    hi = f"{min(yy + int(rng.integers(10, 30)), 99):02d}"
    lo = f"{max(yy - int(rng.integers(10, 19)), 0):02d}"
    ids = tok(txt, add_special_tokens=False).input_ids
    a = tok(hi, add_special_tokens=False).input_ids
    b = tok(lo, add_special_tokens=False).input_ids
    if len(a) != 1 or len(b) != 1:
        return None
    don = [tok(one()[0], add_special_tokens=False).input_ids for _ in range(N_DON)]
    return ids, a[0], b[0], don


def t_agreement(rng, tok, _=None):
    heads = ["keys", "doors", "books", "cables", "reports", "lamps"]
    attrs = ["cabinet", "drawer", "shelf", "office", "machine", "closet"]
    def one():
        return (f"The {heads[int(rng.integers(len(heads)))]} to the "
                f"{attrs[int(rng.integers(len(attrs)))]}")
    ids = tok(one(), add_special_tokens=False).input_ids
    a = tok(" are", add_special_tokens=False).input_ids[0]
    b = tok(" is", add_special_tokens=False).input_ids[0]
    return ids, a, b, [tok(one(), add_special_tokens=False).input_ids for _ in range(N_DON)]


CAPS = [("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"),
        ("Egypt", " Cairo"), ("Cuba", " Havana"), ("Peru", " Lima"),
        ("Norway", " Oslo"), ("Kenya", " Nairobi")]


def t_capital(rng, tok, _=None):
    def one():
        c, _cap = CAPS[int(rng.integers(len(CAPS)))]
        return f"The capital of {c} is", _cap
    txt, cap = one()
    other = CAPS[int(rng.integers(len(CAPS)))][1]
    while other == cap:
        other = CAPS[int(rng.integers(len(CAPS)))][1]
    a = tok(cap, add_special_tokens=False).input_ids
    b = tok(other, add_special_tokens=False).input_ids
    if len(a) != 1 or len(b) != 1:
        return None
    ids = tok(txt, add_special_tokens=False).input_ids
    return ids, a[0], b[0], [tok(one()[0], add_special_tokens=False).input_ids
                             for _ in range(N_DON)]


TASKS = dict(induction=t_induction, greater=t_greater,
             agreement=t_agreement, capital=t_capital)


# ------------------------------------------------------------------ collect ---
def collect(model, tok, modfn, lA, lB, seed, taskfn, bs=8):
    """Byte-for-byte the same sweep structure as adjudicate.collect, task-parameterized."""
    rng = np.random.default_rng(seed)
    got = None
    for _ in range(40):                       # retry only for tokenizer length mismatches
        got = taskfn(rng, tok)
        if got is not None:
            break
    if got is None:
        raise RuntimeError("task could not produce single-token targets")
    base, io, sb, donors = got
    L = len(base)
    donors = [d for d in donors if len(d) == L]          # length-matched, as in IOI
    if len(donors) < 8:
        raise RuntimeError(f"only {len(donors)} length-matched donors")

    ids = torch.tensor([base], device=DEV)
    p = Probe(model, lA, lB, modfn)
    Ac, Bc, _ = batched(model, p, ids, io, sb, bs=1)
    Aclean, Bclean = Ac[0], Bc[0]

    dA, dB = [], []
    for i in range(0, len(donors), 8):
        pid = torch.tensor(donors[i:i + 8], device=DEV)
        p.force = {}
        model(pid)
        dA.append(p.cache["A"].float().cpu().numpy().astype(np.float64))
        dB.append(p.cache["B"].float().cpu().numpy().astype(np.float64))
    dA, dB = np.vstack(dA), np.vstack(dB)

    al = np.linspace(-0.5, 1.5, N_AL)
    T = lambda X: torch.tensor(X, dtype=next(model.parameters()).dtype, device=DEV)
    A1, B1, Y1 = batched(model, p, ids, io, sb,
                         fA=T(np.stack([Aclean + x * (v - Aclean) for v in dA for x in al])),
                         bs=bs)
    A2, B2, Y2 = batched(model, p, ids, io, sb,
                         fB=T(np.stack([Bclean + x * (v - Bclean) for v in dB for x in al])),
                         bs=bs)
    n = len(dA)
    ii, jj = np.meshgrid(np.arange(0, n, 2), np.arange(0, n, 2), indexing="ij")
    A3, B3, Y3 = batched(model, p, ids, io, sb, fA=T(dA[ii.ravel()]),
                         fB=T(dB[jj.ravel()]), bs=bs)
    p.close()

    W = model.lm_head.weight
    u = (W[io] - W[sb]).float().cpu().numpy().astype(np.float64)
    u /= np.linalg.norm(u)
    Zs = np.stack([np.concatenate([A1 @ u, A2 @ u]), np.concatenate([B1 @ u, B2 @ u])], 1)
    Ys = np.concatenate([Y1, Y2])
    Zj = np.stack([A3 @ u, B3 @ u], 1)
    return Zs, Ys, Zj, Y3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--pairs", default="5-7,5-10,6-9,7-9")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpt2", action="store_true")
    ap.add_argument("--out", default=None,
                    help="summary JSON (default nonioi_{task}_{model}.json); use a different "
                         "name when re-capturing a subset of pairs so the frozen summary is kept")
    A = ap.parse_args()
    MTAG = model_tag(A.model)          # 2026-08-27: raw captures are model-tagged (cells.py)
    OUT_JSON = A.out or f"nonioi_{A.task}_{A.model.split('/')[-1]}.json"

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(
        A.model, dtype=torch.bfloat16 if not A.gpt2 else torch.float32,
        device_map=DEV).eval()
    modfn = ((lambda m, l: m.transformer.h[l].attn) if A.gpt2 else
             (lambda m, l: (m.model.layers[l].self_attn if l in FULL_ATTN
                            else m.model.layers[l].linear_attn)))

    pairs = [tuple(int(x) for x in p.split("-")) for p in A.pairs.split(",")]
    seeds = [int(s) for s in A.seeds.split(",")]
    rows, screened = [], 0
    t0 = time.time()
    print(f"=== {A.task}  model={A.model}  FROZEN PROTOCOL, screen std(Yj)>={SCREEN_STD} ===")
    print(f"{'pair':>8} {'seed':>5} {'std(Yj)':>9} {'S-blind':>12} {'J-blind':>12} "
          f"{'BOTH>10x':>9}")
    for (lA, lB) in pairs:
        for sd in seeds:
            try:
                Zs, Ys, Zj, Yj = collect(model, tok, modfn, lA, lB, sd, TASKS[A.task])
            except Exception as ex:
                print(f"{f'{lA}-{lB}':>8} {sd:>5}  ERROR {type(ex).__name__}: {str(ex)[:60]}")
                continue
            sdy = float(np.std(Yj))
            if sdy < SCREEN_STD:
                screened += 1
                print(f"{f'{lA}-{lB}':>8} {sd:>5} {sdy:>9.3f}   -- SCREENED OUT (no signal)")
                continue
            np.savez(raw_name(A.task, MTAG, lA, lB, sd), Zs=Zs, Ys=Ys, Zj=Zj, Yj=Yj)
            r = analyse(Zs, Zj)
            both = (r["sep_Sblind"] > 10) and (r["sep_Jblind"] > 10)
            rows.append(dict(pair=[lA, lB], seed=sd, std_Yj=sdy,
                             sep_S=r["sep_Sblind"], sep_J=r["sep_Jblind"], both=bool(both)))
            print(f"{f'{lA}-{lB}':>8} {sd:>5} {sdy:>9.3f} {r['sep_Sblind']:>12.4g} "
                  f"{r['sep_Jblind']:>12.4g} {str(both):>9}")

    out = dict(task=A.task, model=A.model, screen=SCREEN_STD,
               n_screened_out=screened, n_kept=len(rows), rows=rows)
    if rows:
        sS = np.array([r["sep_S"] for r in rows]); sJ = np.array([r["sep_J"] for r in rows])
        out.update(median_S=float(np.median(sS)), median_J=float(np.median(sJ)),
                   both=int(sum(r["both"] for r in rows)))
        print(f"\n  screened out (no causal signal): {screened}")
        print(f"  median S-blind {np.median(sS):.4g}   median J-blind {np.median(sJ):.4g}")
        print(f"  ** BOTH >10x in {out['both']}/{len(rows)} screened trials **")
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print(f"\nsaved -> {OUT_JSON}  [{time.time()-t0:.0f}s]")
