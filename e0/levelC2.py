"""LEVEL C2 — main runner. Batched, checkpointed, float64 linear algebra.

Protocol (pre-registered):
  DISCOVERY  : approximate nullspace of Phi on Q_single TRAIN only
  FREEZE     : each direction h is frozen; no re-selection downstream
  VALIDATION : same h evaluated on untouched Q_single TEST
  DIAGNOSTIC : same h evaluated on a structurally different family
  CONTROLS   : random directions, shuffled donors, random layer pairs

Reports the FULL distribution over null directions plus a random-direction null,
so a large separation cannot be manufactured by maximising over many h.

Three coordinate systems are always run, because an early audit showed the unembedding
direction explains ~0.01% of activation variance:
   u    : W_U[IO] - W_U[S]
   pc1  : first principal component of the local activation distribution
   diff : mean(donor activations) - clean activation  (task-relevant contrast)
"""
import json, os, time, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
DEV = "cuda:0"
FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31}
kind = lambda l: "F" if l in FULL_ATTN else "D"

TEMPLATES = [
    "When {A} and {B} went to the store, {B} gave a drink to",
    "When {A} and {B} went to the park, {B} handed a ball to",
    "After {A} and {B} finished work, {B} gave the keys to",
    "Because {A} and {B} were hungry, {B} passed the bread to",
]


# ----------------------------------------------------------------- hooks ---
class Probe:
    """Batched read/write of two modules' last-token output."""

    def __init__(s, model, lA, lB, modfn):
        s.cache, s.force, s._h = {}, {}, []
        for nm, l in (("A", lA), ("B", lB)):
            s._h.append(modfn(model, l).register_forward_hook(s._mk(nm)))

    def _mk(s, nm):
        def hook(md, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            f = s.force.get(nm)
            if f is not None:
                o = o.clone()
                o[:, -1, :] = f.to(o.dtype)
                s.cache[nm] = o[:, -1, :].detach().clone()
                return ((o,) + tuple(out[1:])) if isinstance(out, tuple) else o
            s.cache[nm] = o[:, -1, :].detach().clone()
            return out
        return hook

    def close(s):
        for h in s._h:
            h.remove()


def batched(model, probe, ids, io_id, s_id, fA=None, fB=None, bs=16):
    """fA/fB: [N,d] tensors or None. Returns (A_acts[N,d], B_acts[N,d], y[N])."""
    n = (fA if fA is not None else fB).shape[0] if (fA is not None or fB is not None) else 1
    A_out, B_out, Y = [], [], []
    for i in range(0, n, bs):
        j = min(i + bs, n)
        rep = ids.repeat(j - i, 1)
        probe.force = {"A": fA[i:j] if fA is not None else None,
                       "B": fB[i:j] if fB is not None else None}
        lg = model(rep).logits[:, -1, :].float()
        probe.force = {}
        A_out.append(probe.cache["A"].float().cpu())
        B_out.append(probe.cache["B"].float().cpu())
        Y.append((lg[:, io_id] - lg[:, s_id]).cpu())
    return torch.cat(A_out).numpy().astype(np.float64), \
           torch.cat(B_out).numpy().astype(np.float64), \
           torch.cat(Y).numpy().astype(np.float64)


# ------------------------------------------------------------ lin. algebra ---
def exps(d):
    return [(i, t - i) for t in range(d + 1) for i in range(t + 1)]


def design(Z, d):
    e = exps(d)
    return np.array([[a ** i * b ** j for (i, j) in e] for a, b in Z], dtype=np.float64)


def rms(M, h):
    return float(np.linalg.norm(M @ h) / np.sqrt(M.shape[0]))


def nullspace(Mtr, eps):
    cs = np.linalg.norm(Mtr, axis=0); cs[cs == 0] = 1.0
    U, S, Vt = np.linalg.svd(Mtr / cs, full_matrices=True)
    Sn = S / S[0]
    idx = [k for k in range(len(S)) if Sn[k] < eps] + list(range(len(S), Vt.shape[0]))
    H = [(Vt[k] / cs) / np.linalg.norm(Vt[k] / cs) for k in idx]
    return H, Sn, float(S[0] / max(S[-1], 1e-300))


# ------------------------------------------------------------------ main ---
def coords_for(A, B, u, Aclean, Bclean):
    """Return dict name -> (dirA, dirB), each unit-norm float64."""
    out = {}
    un = u / np.linalg.norm(u)
    out["u"] = (un, un)
    for nm, X, cl in (("A", A, Aclean), ("B", B, Bclean)):
        pass
    pcA = np.linalg.svd(A - A.mean(0), full_matrices=False)[2][0]
    pcB = np.linalg.svd(B - B.mean(0), full_matrices=False)[2][0]
    out["pc1"] = (pcA, pcB)
    dA = A.mean(0) - Aclean; dB = B.mean(0) - Bclean
    dA = dA / (np.linalg.norm(dA) + 1e-30); dB = dB / (np.linalg.norm(dB) + 1e-30)
    out["diff"] = (dA, dB)
    return out


def run_pair(model, tok, modfn, lA, lB, seed, names, n_don=32, n_al=7, bs=16,
             degrees=(1, 2, 3), epss=(0.05, 0.02, 0.01, 0.005)):
    rng = np.random.default_rng(seed)
    t = TEMPLATES[rng.integers(len(TEMPLATES))]
    a, b = rng.choice(names, 2, replace=False)
    prompt = t.format(A=a, B=b)
    io_id = tok(" " + a, add_special_tokens=False).input_ids[0]
    s_id = tok(" " + b, add_special_tokens=False).input_ids[0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)

    probe = Probe(model, lA, lB, modfn)
    Ac, Bc, Yc = batched(model, probe, ids, io_id, s_id, bs=1)
    Aclean, Bclean, y_clean = Ac[0], Bc[0], float(Yc[0])

    # donors: DIFFERENT name pairs, same template family
    # donors use the SAME template as the base prompt: names are all single-token,
    # so sequence lengths match by construction (no length filtering needed) and the
    # only thing that varies is the entity content.
    dprompts = []
    for _ in range(n_don):
        x, y = rng.choice(names, 2, replace=False)
        dprompts.append(t.format(A=x, B=y))
    dA, dB = [], []
    L = ids.shape[1]
    same = [p for p in dprompts if tok(p, return_tensors="pt").input_ids.shape[1] == L]
    for i in range(0, len(same), 8):                      # batch equal-length donors
        chunk = same[i:i+8]
        pid = torch.cat([tok(p, return_tensors="pt").input_ids for p in chunk]).to(DEV)
        probe.force = {}
        lg = model(pid).logits[:, -1, :]
        dA.append(probe.cache["A"].float().cpu().numpy().astype(np.float64))
        dB.append(probe.cache["B"].float().cpu().numpy().astype(np.float64))
    dA = np.vstack(dA); dB = np.vstack(dB)
    if len(dA) < 8:
        raise RuntimeError(f"only {len(dA)} length-matched donors")

    n_don = len(dA)
    al = np.linspace(-0.5, 1.5, n_al)
    T = lambda X: torch.tensor(X, dtype=torch.bfloat16, device=DEV)

    # ---- Q_single: patch A only (recompute vs clamped), and patch B only ----
    fA = np.concatenate([Aclean + x * (v - Aclean) for v in dA for x in al])
    fA = fA.reshape(-1, Aclean.shape[0])
    A1r, B1r, Y1r = batched(model, probe, ids, io_id, s_id, fA=T(fA), bs=bs)
    A1c, B1c, Y1c = batched(model, probe, ids, io_id, s_id, fA=T(fA),
                            fB=T(np.tile(Bclean, (fA.shape[0], 1))), bs=bs)
    fB = np.concatenate([Bclean + x * (v - Bclean) for v in dB for x in al])
    fB = fB.reshape(-1, Bclean.shape[0])
    A2, B2, Y2 = batched(model, probe, ids, io_id, s_id, fB=T(fB), bs=bs)

    # ---- Q_joint (diagnostic): INDEPENDENT A and B donors ----
    ii, jj = np.meshgrid(np.arange(0, n_don, 2), np.arange(0, n_don, 2), indexing="ij")
    jA = dA[ii.ravel()]; jB = dB[jj.ravel()]
    A3, B3, Y3 = batched(model, probe, ids, io_id, s_id, fA=T(jA), fB=T(jB), bs=bs)
    probe.close()

    allA = np.vstack([A1r, A1c, A2, A3]); allB = np.vstack([B1r, B1c, B2, B3])
    C = coords_for(allA, allB, (model.lm_head.weight[io_id] - model.lm_head.weight[s_id])
                   .float().cpu().numpy().astype(np.float64), Aclean, Bclean)

    res = {"pair": [int(lA), int(lB)], "type": kind(lA) + "->" + kind(lB), "seed": int(seed),
           "prompt": prompt, "clean_logit_diff": y_clean,
           "n_single_rec": int(len(A1r) + len(A2)), "n_single_cla": int(len(A1c) + len(A2)),
           "n_joint": int(len(A3)), "coords": {}}

    for cname, (da, db) in C.items():
        Zr = np.stack([np.concatenate([A1r @ da, A2 @ da]),
                       np.concatenate([B1r @ db, B2 @ db])], 1)
        Zc = np.stack([np.concatenate([A1c @ da, A2 @ da]),
                       np.concatenate([B1c @ db, B2 @ db])], 1)
        Zj = np.stack([A3 @ da, B3 @ db], 1)
        allZ = np.vstack([Zr, Zc, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
        W = lambda Z: (Z - mu) / sd
        entry = {"range_rec": [float(np.ptp(Zr[:, 0])), float(np.ptp(Zr[:, 1]))],
                 "range_joint": [float(np.ptp(Zj[:, 0])), float(np.ptp(Zj[:, 1]))], "deg": {}}

        for d in degrees:
            idx = rng.permutation(len(Zr)); tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
            Mtr, Mte = design(W(Zr[tr]), d), design(W(Zr[te]), d)
            Mjo = design(W(Zj), d)
            Ctr = design(W(Zc[rng.permutation(len(Zc))[: len(Zc) // 2]]), d)
            per_eps = {}
            for eps in epss:
                H, Sn, cond = nullspace(Mtr, eps)
                rows = [dict(train=rms(Mtr, h), test=rms(Mte, h), diag=rms(Mjo, h),
                             sep=rms(Mjo, h) / max(rms(Mte, h), 1e-12)) for h in H]
                # random-direction null (same basis, same normalisation)
                rnd = []
                for _ in range(120):
                    g = rng.normal(size=Mtr.shape[1]); g /= np.linalg.norm(g)
                    rnd.append(dict(test=rms(Mte, g), diag=rms(Mjo, g),
                                    sep=rms(Mjo, g) / max(rms(Mte, g), 1e-12)))
                Hc, Snc, _ = nullspace(Ctr, eps)
                per_eps[str(eps)] = dict(
                    dimN=len(H), cond=cond, spectrum=[float(x) for x in Sn],
                    dirs=rows, dimN_clamped=len(Hc),
                    rnd_sep_median=float(np.median([r["sep"] for r in rnd])),
                    rnd_sep_p95=float(np.percentile([r["sep"] for r in rnd], 95)),
                    rnd_sep_max=float(np.max([r["sep"] for r in rnd])),
                    rnd_diag_median=float(np.median([r["diag"] for r in rnd])))
            entry["deg"][str(d)] = per_eps
        res["coords"][cname] = entry
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--out", default="c2_qwen.json")
    ap.add_argument("--pairs", default="9-13,21-25,7-11,15-19,5-7,17-19,7-9,19-21")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpt2", action="store_true")
    A = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16,
                                                 device_map=DEV).eval()
    if A.gpt2:
        modfn = lambda m, l: m.transformer.h[l].attn
    else:
        modfn = lambda m, l: (m.model.layers[l].self_attn if l in FULL_ATTN
                              else m.model.layers[l].linear_attn)
    names = [n for n in ["Mary","John","Tom","Sarah","Alice","Bob","Emma","David","Laura",
                         "Peter","Anna","James","Linda","Paul","Susan","Mark","Karen",
                         "Steve","Nancy","Brian","Julia","Kevin","Diana","Frank"]
             if len(tok(" " + n, add_special_tokens=False).input_ids) == 1]
    print(f"[C2] {A.model}  GPU {torch.cuda.memory_allocated()/1e9:.1f} GB  "
          f"{len(names)} single-token names")

    out, done = [], set()
    if os.path.exists(A.out):
        out = json.load(open(A.out))
        done = {(tuple(r["pair"]), r["seed"]) for r in out}
        print(f"[C2] resuming, {len(done)} trials already done")

    pairs = [tuple(int(x) for x in p.split("-")) for p in A.pairs.split(",")]
    seeds = [int(s) for s in A.seeds.split(",")]
    t0 = time.time()
    for (lA, lB) in pairs:
        for sd in seeds:
            if ((lA, lB), sd) in done:
                continue
            try:
                r = run_pair(model, tok, modfn, lA, lB, sd, names)
                out.append(r)
                json.dump(out, open(A.out, "w"))
                e = r["coords"]["u"]["deg"]["3"]["0.02"]
                print(f"[C2] {lA:>2}-{lB:<2} {r['type']} seed{sd} "
                      f"clean={r['clean_logit_diff']:+.2f} dimN={e['dimN']} "
                      f"cond={e['cond']:.1e} "
                      f"maxsep={max([d['sep'] for d in e['dirs']], default=0):.1f} "
                      f"rnd_p95={e['rnd_sep_p95']:.1f}  [{time.time()-t0:.0f}s]")
            except Exception as ex:
                print(f"[C2] {lA}-{lB} seed{sd} ERROR {type(ex).__name__}: {str(ex)[:80]}")
    print(f"[C2] wrote {A.out}  ({len(out)} trials, {time.time()-t0:.0f}s)")
