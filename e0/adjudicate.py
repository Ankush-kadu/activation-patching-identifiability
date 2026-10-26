"""MECHANISM ADJUDICATION — the experiment that makes this mechanistic, not geometric.

Everything so far shows a DIRECTION is near-null on the single-site reachable set.
That is geometry. The mechanistic claim needs more:

  two CONCRETE mechanisms f1, f2 that
    (i)  fit the conventional single-site evidence EQUALLY WELL (held-out),
    (ii) make DIFFERENT predictions under a diagnostic intervention family,
    (iii) and the model's ACTUAL behaviour under that family picks one.

Then conventional activation patching demonstrably did not contain the
information needed to choose between them; the diagnostic family did.

f1 = least-squares fit of Y on the single-site TRAIN set
h  = a near-null direction of the single-site design matrix
f2 = f1 + lam * h      (lam scaled so the joint-family prediction gap is real)
"""
import json, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from levelC2 import Probe, batched, design, rms, nullspace, TEMPLATES, FULL_ATTN, kind, DEV

torch.set_grad_enabled(False)
modfn_qwen = lambda m, l: (m.model.layers[l].self_attn if l in FULL_ATTN
                           else m.model.layers[l].linear_attn)
modfn_gpt2 = lambda m, l: m.transformer.h[l].attn


def collect(model, tok, modfn, lA, lB, seed, names, n_don=32, n_al=7, bs=8):
    rng = np.random.default_rng(seed)
    t = TEMPLATES[rng.integers(len(TEMPLATES))]
    a, b = rng.choice(names, 2, replace=False)
    prompt = t.format(A=a, B=b)
    io = tok(" "+a, add_special_tokens=False).input_ids[0]
    sb = tok(" "+b, add_special_tokens=False).input_ids[0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    p = Probe(model, lA, lB, modfn)
    Ac, Bc, Yc = batched(model, p, ids, io, sb, bs=1)
    Aclean, Bclean = Ac[0], Bc[0]
    dP = [t.format(A=x, B=y) for x, y in
          [rng.choice(names, 2, replace=False) for _ in range(n_don)]]
    dA, dB = [], []
    for i in range(0, len(dP), 8):
        pid = torch.cat([tok(q, return_tensors="pt").input_ids for q in dP[i:i+8]]).to(DEV)
        p.force = {}; model(pid)
        dA.append(p.cache["A"].float().cpu().numpy().astype(np.float64))
        dB.append(p.cache["B"].float().cpu().numpy().astype(np.float64))
    dA, dB = np.vstack(dA), np.vstack(dB)
    al = np.linspace(-0.5, 1.5, n_al)
    T = lambda X: torch.tensor(X, dtype=torch.bfloat16, device=DEV)
    A1, B1, Y1 = batched(model, p, ids, io, sb,
                         fA=T(np.stack([Aclean+x*(v-Aclean) for v in dA for x in al])), bs=bs)
    A2, B2, Y2 = batched(model, p, ids, io, sb,
                         fB=T(np.stack([Bclean+x*(v-Bclean) for v in dB for x in al])), bs=bs)
    n = len(dA)
    ii, jj = np.meshgrid(np.arange(0, n, 2), np.arange(0, n, 2), indexing="ij")
    A3, B3, Y3 = batched(model, p, ids, io, sb, fA=T(dA[ii.ravel()]),
                         fB=T(dB[jj.ravel()]), bs=bs)
    p.close()
    u = (model.lm_head.weight[io]-model.lm_head.weight[sb]).float().cpu().numpy().astype(np.float64)
    u /= np.linalg.norm(u)
    Zs = np.stack([np.concatenate([A1@u, A2@u]), np.concatenate([B1@u, B2@u])], 1)
    Ys = np.concatenate([Y1, Y2])
    Zj = np.stack([A3@u, B3@u], 1); Yj = Y3
    return Zs, Ys, Zj, Yj, prompt, float(Yc[0])


def adjudicate(Zs, Ys, Zj, Yj, deg=3, eps=0.02, seed=0):
    rng = np.random.default_rng(seed)
    allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0)+1e-12
    W = lambda Z: (Z-mu)/sd
    idx = rng.permutation(len(Zs)); tr, te = idx[:len(idx)//2], idx[len(idx)//2:]
    Mtr, Mte, Mjo = design(W(Zs[tr]), deg), design(W(Zs[te]), deg), design(W(Zj), deg)
    ytr, yte = Ys[tr], Ys[te]

    f1, *_ = np.linalg.lstsq(Mtr, ytr, rcond=None)
    H, Sn, cond = nullspace(Mtr, eps)
    if not H:
        return None
    # pick the null direction with the LARGEST joint-family effect, chosen on
    # TRAIN-derived quantities only (Mjo is intervention geometry, not held-out Y)
    h = max(H, key=lambda v: rms(Mjo, v))
    # scale so f2 differs from f1 by ~1 logit RMS on the joint family
    lam = 1.0 / max(rms(Mjo, h), 1e-12)

    f2 = f1 + lam*h
    err1_te = float(np.sqrt(np.mean((Mte@f1 - yte)**2)))
    err2_te = float(np.sqrt(np.mean((Mte@f2 - yte)**2)))
    pred1_j, pred2_j = Mjo@f1, Mjo@f2
    err1_j = float(np.sqrt(np.mean((pred1_j - Yj)**2)))
    err2_j = float(np.sqrt(np.mean((pred2_j - Yj)**2)))
    gap_j = float(np.sqrt(np.mean((pred1_j - pred2_j)**2)))
    ystd = float(np.std(Yj))
    return dict(cond=cond, dimN=len(H),
                single_test_err_f1=err1_te, single_test_err_f2=err2_te,
                single_test_err_ratio=err2_te/max(err1_te, 1e-12),
                joint_pred_gap=gap_j, joint_err_f1=err1_j, joint_err_f2=err2_j,
                joint_Y_std=ystd, n_single=len(Zs), n_joint=len(Zj),
                verdict=("ADJUDICATED" if (err2_te/max(err1_te,1e-12) < 1.10
                                           and err2_j > 1.5*err1_j) else "no"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--gpt2", action="store_true")
    ap.add_argument("--pairs", default="9-13,7-11,5-7,7-9")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="adjudicate_qwen.json")
    A = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16,
                                                 device_map=DEV).eval()
    modfn = modfn_gpt2 if A.gpt2 else modfn_qwen
    names = [n for n in ["Mary","John","Tom","Sarah","Alice","Bob","Emma","David",
                         "Laura","Peter","Anna","James","Linda","Paul","Susan","Mark"]
             if len(tok(" "+n, add_special_tokens=False).input_ids) == 1]
    OUT = []
    print(f"{'pair':>8} {'sd':>3} {'dimN':>5} {'f1 test':>9} {'f2 test':>9} {'ratio':>7} "
          f"{'joint gap':>10} {'f1 jerr':>8} {'f2 jerr':>8} {'Y std':>7}  verdict")
    print("-"*104)
    for pr in A.pairs.split(","):
        lA, lB = (int(x) for x in pr.split("-"))
        for sd in (int(s) for s in A.seeds.split(",")):
            try:
                Zs, Ys, Zj, Yj, prompt, y0 = collect(model, tok, modfn, lA, lB, sd, names)
                np.savez(f"adj_raw_{lA}_{lB}_{sd}.npz", Zs=Zs, Ys=Ys, Zj=Zj, Yj=Yj)
                r = adjudicate(Zs, Ys, Zj, Yj)
                if r is None:
                    print(f"{lA:>3}-{lB:<4} {sd:>3}   no null direction"); continue
                r.update(pair=[lA, lB], seed=sd, prompt=prompt, clean=y0)
                OUT.append(r); json.dump(OUT, open(A.out, "w"), indent=1)
                print(f"{lA:>3}-{lB:<4} {sd:>3} {r['dimN']:>5} {r['single_test_err_f1']:>9.4f} "
                      f"{r['single_test_err_f2']:>9.4f} {r['single_test_err_ratio']:>7.3f} "
                      f"{r['joint_pred_gap']:>10.3f} {r['joint_err_f1']:>8.3f} "
                      f"{r['joint_err_f2']:>8.3f} {r['joint_Y_std']:>7.3f}  {r['verdict']}")
            except Exception as ex:
                print(f"{lA:>3}-{lB:<4} {sd:>3} ERROR {type(ex).__name__}: {str(ex)[:60]}")
    print(f"\nwrote {A.out}")
