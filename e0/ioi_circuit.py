"""NAMED PUBLISHED CIRCUIT: does single-site evidence determine the IOI mechanism?

Wang et al. (arXiv:2211.00593) identify a 26-head GPT-2-small IOI circuit in which
S-INHIBITION heads modulate NAME MOVER heads (writing to the Name Movers' query so
they attend away from S). That published claim is itself an INTERACTION claim:
the Name Mover's contribution is asserted to DEPEND ON the S-Inhibition state.

So the two mechanistic hypotheses are not invented for this paper:
  H_add  : S-Inhibition and Name Mover contribute INDEPENDENTLY (additive)
  H_int  : the Name Mover's contribution DEPENDS ON the S-Inhibition state

Question: does conventional single-site activation patching DETERMINE which holds?

Per-head patching is done by a forward_pre_hook on attn.c_proj, editing the
head's own 64-dim slice of the concatenated head outputs (pre-W_O).
"""
import json, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_grad_enabled(False)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
NH, HD = 12, 64

# published IOI head classes (Wang et al. 2211.00593), GPT-2 small
S_INHIB   = [(7,3),(7,9),(8,6),(8,10)]
NAME_MOVE = [(9,9),(9,6),(10,0)]
NEG_NM    = [(10,7),(11,10)]
INDUCTION = [(5,5),(5,8),(5,9),(6,9)]

class HeadProbe:
    """Read/patch a single attention head's pre-W_O output at the last position."""
    def __init__(s, model, heads):
        s.cache, s.force, s._h = {}, {}, []
        s.heads = heads
        for (l, h) in heads:
            mod = model.transformer.h[l].attn.c_proj
            s._h.append(mod.register_forward_pre_hook(s._mk(l, h)))
    def _mk(s, l, h):
        sl = slice(h*HD, (h+1)*HD)
        def pre(mod, args):
            x = args[0]
            f = s.force.get((l, h))
            if f is not None:
                x = x.clone(); x[:, -1, sl] = f.to(x.dtype)
            s.cache[(l, h)] = x[:, -1, sl].detach().clone()
            return (x,) + tuple(args[1:])
        return pre
    def close(s):
        for h in s._h: h.remove()

def head_dir(model, l, h, u):
    """Direction in the head's 64-dim space whose projection = contribution to u."""
    W = model.transformer.h[l].attn.c_proj.weight        # [768, 768] (in, out)
    return (W[h*HD:(h+1)*HD, :].float() @ u)

TEMPLATES = ["When {A} and {B} went to the store, {B} gave a drink to",
             "When {A} and {B} went to the park, {B} handed a ball to",
             "After {A} and {B} finished work, {B} gave the keys to"]
NAMES = ["Mary","John","Tom","Sarah","Alice","Bob","Emma","David","Laura","Peter","Anna","James"]

def collect(model, tok, hA, hB, seed=0, n_don=32, n_al=7, bs=16):
    rng = np.random.default_rng(seed)
    t = TEMPLATES[rng.integers(len(TEMPLATES))]
    a, b = rng.choice(NAMES, 2, replace=False)
    prompt = t.format(A=a, B=b)
    io = tok(" "+a).input_ids[0]; sb = tok(" "+b).input_ids[0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    p = HeadProbe(model, [hA, hB])
    u = (model.lm_head.weight[io] - model.lm_head.weight[sb]).float()
    dirA, dirB = head_dir(model, *hA, u), head_dir(model, *hB, u)

    def fwd(fA=None, fB=None, n=1):
        p.force = {}
        if fA is not None: p.force[hA] = fA
        if fB is not None: p.force[hB] = fB
        rep = ids.repeat(n, 1)
        lg = model(rep).logits[:, -1, :].float(); p.force = {}
        za, zb = p.cache[hA].float(), p.cache[hB].float()
        return ((za @ dirA).cpu().numpy().astype(np.float64),
                (zb @ dirB).cpu().numpy().astype(np.float64),
                (lg[:, io] - lg[:, sb]).cpu().numpy().astype(np.float64))
    a0, b0, y0 = fwd()
    zA_clean, zB_clean = p.cache[hA][0].clone(), p.cache[hB][0].clone()

    # donors: same template, different names
    dA, dB = [], []
    dP = [t.format(A=x, B=y) for x, y in [rng.choice(NAMES,2,replace=False) for _ in range(n_don)]]
    for i in range(0, len(dP), 8):
        pid = torch.cat([tok(q, return_tensors="pt").input_ids for q in dP[i:i+8]]).to(DEV)
        p.force = {}; model(pid)
        dA.append(p.cache[hA].clone()); dB.append(p.cache[hB].clone())
    dA, dB = torch.cat(dA), torch.cat(dB)

    al = np.linspace(-0.5, 1.5, n_al)
    def sweep(kind):
        A_,B_,Y_ = [],[],[]
        if kind != "joint":
            for v in dA:
                f = torch.stack([zA_clean + x*(v-zA_clean) for x in al])
                r = fwd(fA=f, n=len(al)); A_.append(r[0]); B_.append(r[1]); Y_.append(r[2])
            for v in dB:
                f = torch.stack([zB_clean + x*(v-zB_clean) for x in al])
                r = fwd(fB=f, n=len(al)); A_.append(r[0]); B_.append(r[1]); Y_.append(r[2])
        else:
            n = len(dA)
            for i in range(0, n, 2):
                fA = dA[i].unsqueeze(0).repeat(len(range(0,n,2)),1)
                fB = torch.stack([dB[j] for j in range(0,n,2)])
                r = fwd(fA=fA, fB=fB, n=fB.shape[0]); A_.append(r[0]); B_.append(r[1]); Y_.append(r[2])
        return (np.concatenate(A_), np.concatenate(B_), np.concatenate(Y_))
    As, Bs, Ys = sweep("single"); Aj, Bj, Yj = sweep("joint")
    p.close()
    return np.stack([As,Bs],1), Ys, np.stack([Aj,Bj],1), Yj, prompt, float(y0[0])

if __name__ == "__main__":
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    ADD=[(0,0),(1,0),(0,1),(2,0),(0,2),(3,0),(0,3)]
    FULL=[(i,t-i) for t in range(4) for i in range(t+1)]
    CI=[FULL.index(e) for e in FULL if e not in ADD]
    dm=lambda Z,E: np.array([[a**i*b**j for (i,j) in E] for a,b in Z],float)
    ridge=lambda M,y,lam: np.linalg.solve(M.T@M+lam*np.eye(M.shape[1]), M.T@y)

    PAIRS=[(("S-Inhib","NameMover"),(8,6),(9,9)), (("S-Inhib","NameMover"),(7,9),(9,6)),
           (("S-Inhib","NameMover"),(8,10),(10,0)), (("Induction","NameMover"),(5,5),(9,9)),
           (("NameMover","NegNameMover"),(9,9),(10,7))]
    print(f"{'published pair':>26} {'heads':>14} {'sd':>3} {'std Yj':>7} | "
          f"{'|c_x| SINGLE':>13} {'JOINT':>9} {'ratio':>8} {'ridge':>8} | {'S-fit RMSE add/int':>19}")
    print("-"*116)
    OUT=[]
    for (nm, hA, hB) in PAIRS:
        for sd in (0,1,2):
            try:
                Zs,Ys,Zj,Yj,prompt,y0 = collect(model,tok,hA,hB,sd)
                mu,s=np.vstack([Zs,Zj]).mean(0),np.vstack([Zs,Zj]).std(0)+1e-12
                W=lambda Z:(Z-mu)/s
                Ms,Mj=dm(W(Zs),FULL),dm(W(Zj),FULL)
                cs=np.linalg.lstsq(Ms,Ys,rcond=None)[0]; cj=np.linalg.lstsq(Mj,Yj,rcond=None)[0]
                ns,nj=np.linalg.norm(cs[CI]),np.linalg.norm(cj[CI])
                nsr=np.linalg.norm(ridge(Ms,Ys,1e-2)[CI])
                # additive vs full on held-out single
                rng=np.random.default_rng(sd); i=rng.permutation(len(Zs)); tr,te=i[:len(i)//2],i[len(i)//2:]
                ca=np.linalg.lstsq(dm(W(Zs[tr]),ADD),Ys[tr],rcond=None)[0]
                cf=np.linalg.lstsq(dm(W(Zs[tr]),FULL),Ys[tr],rcond=None)[0]
                ea=float(np.sqrt(np.mean((dm(W(Zs[te]),ADD)@ca-Ys[te])**2)))
                ef=float(np.sqrt(np.mean((dm(W(Zs[te]),FULL)@cf-Ys[te])**2)))
                r=dict(pair=nm,hA=list(hA),hB=list(hB),seed=sd,std_Yj=float(np.std(Yj)),
                       cross_single=float(ns),cross_joint=float(nj),cross_ridge=float(nsr),
                       ratio=float(ns/max(nj,1e-12)),ridge_ratio=float(nsr/max(nj,1e-12)),
                       rmse_add=ea,rmse_full=ef,prompt=prompt,clean=y0)
                OUT.append(r); json.dump(OUT,open("ioi_circuit.json","w"),indent=1)
                print(f"{nm[0]+'->'+nm[1]:>26} {str(hA)+'->'+str(hB):>14} {sd:>3} {r['std_Yj']:>7.3f} | "
                      f"{ns:>13.4f} {nj:>9.4f} {r['ratio']:>7.0f}x {r['ridge_ratio']:>7.1f}x | "
                      f"{ea:>9.4f}/{ef:<9.4f}")
            except Exception as e:
                print(f"{nm[0]+'->'+nm[1]:>26} {str(hA):>14} {sd:>3} ERR {type(e).__name__}: {str(e)[:40]}")
    a=np.array([[r['ratio'],r['ridge_ratio'],r['rmse_add'],r['rmse_full']] for r in OUT])
    print(f"\nMEDIAN ratio {np.median(a[:,0]):.0f}x   ridge {np.median(a[:,1]):.1f}x   "
          f"additive/full held-out RMSE {np.median(a[:,2]):.4f} / {np.median(a[:,3]):.4f} "
          f"(ratio {np.median(a[:,3])/np.median(a[:,2]):.3f})")
