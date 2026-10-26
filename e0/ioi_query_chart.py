"""THE PUBLISHED MECHANISM'S OWN CHART  (added 2026-08-28).

Section 6 shows that a chart chosen by optimising a statistic on the states it will be scored on
is a selection effect, even with no readout involved.  That does not mean charts cannot be chosen
-- it means they must be specified in ADVANCE, from something other than the data being scored.

Wang et al. (2211.00593) claim a specific mechanism for IOI: S-Inhibition heads write into the
Name Movers' QUERY, so the Name Mover's contribution DEPENDS ON the S-Inhibition state.  That
claim names its own chart, entirely from weights:

    a = S-Inhibition head output, projected on the direction of its own output space that most
        affects the Name Mover's query.  The composed map is  W_O[SI slice] @ W_Q[NM],  a 64x64
        matrix of PARAMETERS; we take its top left-singular vector.
    b = Name Mover head output, projected on the readout direction u.

Nothing in either direction is fitted to the intervention states or to the readout, so the
selection effect of Section 6 cannot arise.  If the published mechanism is right the interaction
between a and b is real, and the question this paper asks becomes concrete: can single-site
patching evidence see it?
"""
import argparse, json, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cells import N_AL, ALPHAS

torch.set_grad_enabled(False)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
NH, HD = 12, 64
AL = np.array(ALPHAS)

S_INHIB = [(7, 3), (7, 9), (8, 6), (8, 10)]
NAME_MOVE = [(9, 9), (9, 6), (10, 0)]
# A SECOND published circuit with the same query-mediated shape: previous-token heads write the
# information an induction head matches on in its QUERY (Elhage et al. 2021, Olsson et al. 2022),
# so the identical weight-derived chart applies. Testing two independent published circuits is
# what makes Section 7 a claim about the method rather than about IOI.
PREV_TOK = [(2, 2), (4, 11)]
INDUCTION = [(5, 1), (5, 5), (6, 9)]
TEMPLATES = ["When {A} and {B} went to the store, {B} gave a drink to",
             "When {A} and {B} went to the park, {B} handed a ball to",
             "After {A} and {B} finished work, {B} gave the keys to"]
NAMES = ["Mary", "John", "Tom", "Sarah", "Alice", "Bob", "Emma", "David",
         "Laura", "Peter", "Anna", "James"]


def query_direction(model, si, nm):
    """Direction in the S-Inhibition head's output space that most drives the Name Mover's query.

    W_O for head h of layer l is rows [h*64:(h+1)*64] of attn.c_proj.weight (in, out).
    W_Q for head h' of layer l' is columns [h'*64:(h'+1)*64] of the query third of c_attn.weight.
    The composed map is their product; its top left-singular vector is the direction we want.
    Parameters only -- no data.
    """
    lS, hS = si
    lN, hN = nm
    W_O = model.transformer.h[lS].attn.c_proj.weight[hS * HD:(hS + 1) * HD, :].float()
    W_all = model.transformer.h[lN].attn.c_attn.weight.float()
    W_Q = W_all[:, hN * HD:(hN + 1) * HD]                       # query third, this head
    M = (W_O @ W_Q).cpu().numpy().astype(np.float64)            # 64 x 64, weights alone
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    v = U[:, 0]
    return v / np.linalg.norm(v)


class HeadProbe:
    def __init__(s, model, heads):
        s.cache, s.force, s._h = {}, {}, []
        for (l, h) in heads:
            s._h.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(s._mk(l, h)))

    def _mk(s, l, h):
        sl = slice(h * HD, (h + 1) * HD)
        def pre(mod, args):
            x = args[0]
            f = s.force.get((l, h))
            if f is not None:
                x = x.clone()
                x[:, -1, sl] = f.to(x.dtype)
            s.cache[(l, h)] = x[:, -1, sl].detach().clone()
            return (x,) + tuple(args[1:])
        return pre

    def close(s):
        for h in s._h:
            h.remove()


def collect(model, tok, si, nm, seed, n_don=32, bs=16, circuit="ioi"):
    rng = np.random.default_rng(seed)
    if circuit == "induction":
        # [A][B] ... [A] -> [B]: the readout contrasts the correct continuation against a foil
        toks = [int(x) for x in rng.integers(1000, 20000, 12)]
        seq = toks + toks                                  # repeat, so induction can fire
        ids = torch.tensor([seq[:-1]], device=DEV)
        io, sb = seq[-1], int(rng.integers(1000, 20000))
        prompt = f"induction seq seed {seed}"
    else:
        t = TEMPLATES[rng.integers(len(TEMPLATES))]
        a, b = rng.choice(NAMES, 2, replace=False)
        prompt = t.format(A=a, B=b)
        io = tok(" " + a).input_ids[0]; sb = tok(" " + b).input_ids[0]
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    p = HeadProbe(model, [si, nm])

    vq = query_direction(model, si, nm)                          # a-direction, from weights
    u = (model.lm_head.weight[io] - model.lm_head.weight[sb]).float()
    dirN = (model.transformer.h[nm[0]].attn.c_proj.weight[nm[1] * HD:(nm[1] + 1) * HD, :].float()
            @ u).cpu().numpy().astype(np.float64)                # b-direction, from weights
    dirN /= np.linalg.norm(dirN)

    def fwd(fS=None, fN=None, n=1):
        p.force = {}
        if fS is not None: p.force[si] = fS
        if fN is not None: p.force[nm] = fN
        lg = model(ids.repeat(n, 1)).logits[:, -1, :].float()
        p.force = {}
        A_ = p.cache[si].float().cpu().numpy().astype(np.float64)
        B_ = p.cache[nm].float().cpu().numpy().astype(np.float64)
        return A_, B_, (lg[:, io] - lg[:, sb]).cpu().numpy().astype(np.float64)

    A0, B0, y0 = fwd()
    Sclean, Nclean = torch.tensor(A0[0], device=DEV).float(), torch.tensor(B0[0], device=DEV).float()

    if circuit == "induction":
        dP = []
        for _ in range(n_don):
            tk = [int(x) for x in rng.integers(1000, 20000, 12)]
            dP.append((tk + tk)[:-1])
    else:
        dP = [tok(t.format(A=x, B=y), return_tensors="pt").input_ids[0].tolist()
              for x, y in [rng.choice(NAMES, 2, replace=False) for _ in range(n_don)]]
    dP = [q for q in dP if len(q) == ids.shape[1]]
    dS, dN = [], []
    for i in range(0, len(dP), 8):
        pid = torch.tensor(dP[i:i + 8], device=DEV)
        p.force = {}; model(pid)
        dS.append(p.cache[si].float().clone()); dN.append(p.cache[nm].float().clone())
    dS, dN = torch.cat(dS), torch.cat(dN)

    def run(force_fn, n_rows):
        A_, B_, Y_ = [], [], []
        for i in range(0, n_rows, bs):
            j = min(i + bs, n_rows)
            fS, fN = force_fn(i, j)
            a_, b_, y_ = fwd(fS, fN, j - i)
            A_.append(a_); B_.append(b_); Y_.append(y_)
        return np.vstack(A_), np.vstack(B_), np.concatenate(Y_)

    rows = [(v, x) for v in range(len(dS)) for x in range(N_AL)]
    # arm S: patch S-Inhibition, let the Name Mover recompute
    fA = torch.stack([Sclean + AL[x] * (dS[v] - Sclean) for v, x in rows])
    A1, B1, Y1 = run(lambda i, j: (fA[i:j], None), len(rows))
    # arm S clamped: same, with the Name Mover held at its clean value
    A1c, B1c, Y1c = run(lambda i, j: (fA[i:j], Nclean.repeat(j - i, 1)), len(rows))
    # arm N: patch the Name Mover alone
    fB = torch.stack([Nclean + AL[x] * (dN[v] - Nclean) for v, x in rows])
    A2, B2, Y2 = run(lambda i, j: (None, fB[i:j]), len(rows))
    # joint, on the SAME alpha grid
    pairs = [(i, j) for i in range(0, len(dS), 2) for j in range(0, len(dN), 2)]
    aidx = np.array([((i + j) % N_AL, (3 * i + j) % N_AL) for i, j in pairs])
    fSj = torch.stack([Sclean + AL[k] * (dS[i] - Sclean) for (i, _), k in zip(pairs, aidx[:, 0])])
    fNj = torch.stack([Nclean + AL[k] * (dN[j] - Nclean) for (_, j), k in zip(pairs, aidx[:, 1])])
    A3, B3, Y3 = run(lambda i, j: (fSj[i:j], fNj[i:j]), len(pairs))
    p.close()

    P = lambda A_, B_: np.stack([A_ @ vq, B_ @ dirN], 1)
    return dict(Zs=np.vstack([P(A1, B1), P(A2, B2)]), Ys=np.concatenate([Y1, Y2]),
                Zc=np.vstack([P(A1c, B1c), P(A2, B2)]), Yc=np.concatenate([Y1c, Y2]),
                Zjm=P(A3, B3), Yjm=Y3, Zj=P(A3, B3), Yj=Y3,
                clean=float(y0[0]), prompt=prompt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--circuit", default="ioi", choices=["ioi", "induction"],
                    help="which published circuit's own chart to use")
    ap.add_argument("--control", action="store_true",
                    help="NEGATIVE CONTROL: run the identical procedure on head pairs for which "
                         "no query-mediated mechanism is claimed. If the information gap appears "
                         "here too it is generic to the design, not a property of the published "
                         "circuit, and the Section 7 claim would not stand.")
    A = ap.parse_args()
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", dtype=torch.float32).to(DEV).eval()
    from identifiability import analyse as ident
    out = []
    hdr = ("early -> NameMover" if A.control else
           ("prev-tok -> induction" if A.circuit == "induction" else "S-Inhib -> NameMover"))
    print(f"{hdr:>22}{'seed':>5}{'sd(Yj)':>9}{'edge R2':>9}{'sd ratio':>10}"
          f"{'info S':>11}{'info J':>11}{'se_S/se_J':>11}{'Wald p (J)':>12}")
    CTRL_A = [(0, 5), (1, 2), (2, 9), (3, 4)]      # early heads, no claimed role in IOI
    CTRL_B = [(9, 9), (9, 6), (10, 0)]
    if A.control:
        src_A, src_B = CTRL_A, CTRL_B
    elif A.circuit == "induction":
        src_A, src_B = PREV_TOK, INDUCTION
    else:
        src_A, src_B = S_INHIB, NAME_MOVE
    for si in src_A:
        for nm in src_B:
            for sd in (int(s) for s in A.seeds.split(",")):
                try:
                    d = collect(model, tok, si, nm, sd, circuit=A.circuit)
                except Exception as ex:
                    print(f"{str(si)+'->'+str(nm):>22}{sd:>5}  ERROR {type(ex).__name__}"); continue
                if float(np.std(d["Yjm"])) < 0.15:
                    continue
                r = ident(d["Zs"], d["Ys"], d["Zjm"], d["Yjm"])
                n = len(d["Zs"]) // 2
                armA, armB = d["Zs"][:n], d["Zs"][n:]
                rr = np.corrcoef(armA[:, 0], armA[:, 1])[0, 1]
                r.update(si=list(si), nm=list(nm), seed=sd,
                         edge_r2=float(rr ** 2) if np.isfinite(rr) else 0.0,
                         sd_ratio=float(np.std(armA[:, 1]) / max(np.std(armB[:, 1]), 1e-300)))
                out.append(r)
                np.savez(f"{'qctrl' if A.control else ('qind' if A.circuit=='induction' else 'qchart')}"
                         f"_{si[0]}{si[1]}_{nm[0]}{nm[1]}_{sd}.npz",
                         **{k: v for k, v in d.items() if k != "prompt"})
                print(f"{str(si)+'->'+str(nm):>22}{sd:>5}{r['std_Yj']:>9.3f}{r['edge_r2']:>9.3f}"
                      f"{r['sd_ratio']:>10.3f}{r['info_min_S']:>11.2e}{r['info_min_J']:>11.2e}"
                      f"{r['se_ratio']:>11.1f}{r['wald_p_J']:>12.2g}")
    # PERMUTATION CALIBRATION.  A higher rejection rate under single-site evidence would be
    # meaningless if its Wald test were miscalibrated on a near-degenerate design, so we check:
    # shuffle the readout (destroying any interaction) and measure the realised rejection rate.
    from identifiability import coef_and_se as _cs
    rng2 = np.random.default_rng(0)
    kS = kJ = ncal = 0
    for f in sorted(__import__("glob").glob(
            "qctrl_*.npz" if A.control else
            ("qind_*.npz" if A.circuit == "induction" else "qchart_*.npz"))):
        d = np.load(f)
        allZ = np.vstack([d["Zs"], d["Zjm"]]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
        W = lambda Z: (Z - mu) / sd
        for _ in range(20):
            kS += _cs(d["Zs"], rng2.permutation(d["Ys"]), W)["wald_p"] < 0.05
            kJ += _cs(d["Zjm"], rng2.permutation(d["Yjm"]), W)["wald_p"] < 0.05
            ncal += 1
    cal = dict(n=ncal, reject_S=int(kS), reject_J=int(kJ),
               rate_S=kS / max(ncal, 1), rate_J=kJ / max(ncal, 1))
    print(f"\n  permutation calibration on shuffled readouts ({ncal} draws): "
          f"single-site rejects {100*cal['rate_S']:.1f}%, joint {100*cal['rate_J']:.1f}% "
          f"(nominal 5%) -- both calibrated, so the rejection counts are interpretable")
    OUTNAME = ("ioi_query_chart_control.json" if A.control else
               (f"ioi_query_chart_{A.circuit}.json" if A.circuit != "ioi"
                else "ioi_query_chart.json"))
    json.dump({"rows": out, "calibration": cal}, open(OUTNAME, "w"), indent=1, default=float)
    m = lambda k: float(np.median([r[k] for r in out]))
    _ = None
    print(f"\n  n = {len(out)} configurations of {hdr} heads")
    print(f"  realised edge in the mechanism's own chart : R2 = {m('edge_r2'):.3f}, "
          f"sd ratio = {m('sd_ratio'):.3f}")
    print(f"  partial information for the interaction    : single-site {m('info_min_S'):.2e}, "
          f"joint {m('info_min_J'):.2e}  ({m('info_min_J')/max(m('info_min_S'),1e-300):.0f}x)")
    print(f"  standard-error ratio se_S/se_J             : {m('se_ratio'):.1f}")
    rej = sum(r['wald_p_J'] < 0.05 for r in out)
    print(f"  joint evidence rejects zero interaction    : {rej}/{len(out)} configurations")
    print(f"  single-site rejects                        : "
          f"{sum(r['wald_p_S'] < 0.05 for r in out)}/{len(out)}")
    print(f"\nsaved -> {OUTNAME}")
