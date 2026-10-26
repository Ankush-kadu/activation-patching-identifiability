"""SUBMISSION-CRITICAL: does the RECIPROCAL blind-subspace result survive a change
of STATE COORDINATE, not just a change of function basis?

The headline reciprocal test so far used ONE coordinate: the unembedding direction
u = W_U[IO]-W_U[S]. an early audit showed u carries ~0.01% of activation variance, so the
result must be shown not to depend on that chart. Basis robustness (poly/RBF/
Fourier) is a DIFFERENT axis and does not address this.

Coordinates tested:
  u     unembedding logit-difference direction
  pc1   first principal component of the component's own activation distribution
  diff  mean(donor) - clean  (task contrast direction)
  2D    a genuinely MULTIDIMENSIONAL chart: top-2 PCs per component (4 features)
"""
import json, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.linalg import eigh
torch.set_grad_enabled(False)
DEV="cuda:0" if torch.cuda.is_available() else "cpu"
TEMPLATES=["When {A} and {B} went to the store, {B} gave a drink to",
           "When {A} and {B} went to the park, {B} handed a ball to",
           "After {A} and {B} finished work, {B} gave the keys to"]
NAMES=["Mary","John","Tom","Sarah","Alice","Bob","Emma","David","Laura","Peter","Anna","James"]

class P:
    def __init__(s,m,lA,lB):
        s.c,s.f,s._h={},{},[]
        for nm,l in (("A",lA),("B",lB)):
            s._h.append(m.transformer.h[l].attn.register_forward_hook(s._mk(nm)))
    def _mk(s,nm):
        def h(md,i,o):
            x=o[0] if isinstance(o,tuple) else o
            g=s.f.get(nm)
            if g is not None:
                x=x.clone(); x[:,-1,:]=g.to(x.dtype)
                s.c[nm]=x[:,-1,:].detach().clone()
                return ((x,)+tuple(o[1:])) if isinstance(o,tuple) else x
            s.c[nm]=x[:,-1,:].detach().clone(); return o
        return h
    def close(s):
        for h in s._h: h.remove()

def collect_full(model,tok,lA,lB,seed,n_don=28,n_al=7,bs=14):
    """Return FULL activation arrays so coordinates can be chosen post hoc."""
    rng=np.random.default_rng(seed); t=TEMPLATES[rng.integers(len(TEMPLATES))]
    a,b=rng.choice(NAMES,2,replace=False); prompt=t.format(A=a,B=b)
    io=tok(" "+a).input_ids[0]; sb=tok(" "+b).input_ids[0]
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    p=P(model,lA,lB)
    def fwd(fA=None,fB=None,n=1):
        p.f={}
        if fA is not None: p.f["A"]=fA
        if fB is not None: p.f["B"]=fB
        lg=model(ids.repeat(n,1)).logits[:,-1,:].float(); p.f={}
        return (p.c["A"].float().cpu().numpy().astype(np.float64),
                p.c["B"].float().cpu().numpy().astype(np.float64),
                (lg[:,io]-lg[:,sb]).cpu().numpy().astype(np.float64))
    Ac,Bc,_=fwd(); zA,zB=torch.tensor(Ac[0]).to(DEV),torch.tensor(Bc[0]).to(DEV)
    dA,dB=[],[]
    dP=[t.format(A=x,B=y) for x,y in [rng.choice(NAMES,2,replace=False) for _ in range(n_don)]]
    for q in dP:
        p.f={}; model(tok(q,return_tensors="pt").input_ids.to(DEV))
        dA.append(p.c["A"][0].clone()); dB.append(p.c["B"][0].clone())
    dA,dB=torch.stack(dA),torch.stack(dB)
    al=np.linspace(-0.5,1.5,n_al)
    SA,SB=[],[]
    for v in dA:
        f=torch.stack([zA+x*(v-zA) for x in al]); r=fwd(fA=f,n=len(al)); SA.append(r[0]); SB.append(r[1])
    for v in dB:
        f=torch.stack([zB+x*(v-zB) for x in al]); r=fwd(fB=f,n=len(al)); SA.append(r[0]); SB.append(r[1])
    JA,JB=[],[]
    idx=list(range(0,len(dA),2))
    for i in idx:
        fB=torch.stack([dB[j] for j in idx]); fA=dA[i].unsqueeze(0).repeat(len(idx),1)
        r=fwd(fA=fA,fB=fB,n=len(idx)); JA.append(r[0]); JB.append(r[1])
    p.close()
    u=(model.lm_head.weight[io]-model.lm_head.weight[sb]).float().cpu().numpy().astype(np.float64)
    return (np.vstack(SA),np.vstack(SB),np.vstack(JA),np.vstack(JB),
            Ac[0],Bc[0],u/np.linalg.norm(u))

def coords(SA,SB,JA,JB,Acl,Bcl,u,kind):
    allA=np.vstack([SA,JA]); allB=np.vstack([SB,JB])
    if kind=="u":    da,db=u,u
    elif kind=="pc1":
        da=np.linalg.svd(allA-allA.mean(0),full_matrices=False)[2][0]
        db=np.linalg.svd(allB-allB.mean(0),full_matrices=False)[2][0]
    elif kind=="diff":
        da=allA.mean(0)-Acl; db=allB.mean(0)-Bcl
        da/=np.linalg.norm(da)+1e-30; db/=np.linalg.norm(db)+1e-30
    elif kind=="pc2d":
        Va=np.linalg.svd(allA-allA.mean(0),full_matrices=False)[2][:2]
        Vb=np.linalg.svd(allB-allB.mean(0),full_matrices=False)[2][:2]
        return (np.hstack([SA@Va.T,SB@Vb.T]), np.hstack([JA@Va.T,JB@Vb.T]))
    return (np.stack([SA@da,SB@db],1), np.stack([JA@da,JB@db],1))

def design(Z,d=3):
    k=Z.shape[1]
    if k==2:
        E=[(i,t-i) for t in range(d+1) for i in range(t+1)]
        return np.array([[a**i*b**j for (i,j) in E] for a,b in Z],float)
    # 4-D chart: linear + all pairwise products + squares (degree 2, 15 terms)
    out=[np.ones(len(Z))]+[Z[:,i] for i in range(k)]
    for i in range(k):
        for j in range(i,k): out.append(Z[:,i]*Z[:,j])
    return np.stack(out,1)

def recip(Za,Zb,seed=0,lam=1e-10):
    allZ=np.vstack([Za,Zb]); mu,sd=allZ.mean(0),allZ.std(0)+1e-12
    W=lambda Z:(Z-mu)/sd
    cs=np.linalg.norm(design(W(allZ)),axis=0); cs[cs==0]=1
    Ma,Mb=design(W(Za))/cs,design(W(Zb))/cs
    rng=np.random.default_rng(seed); i=rng.permutation(len(Ma)); j=rng.permutation(len(Mb))
    Atr,Ate=Ma[i[:len(i)//2]],Ma[i[len(i)//2:]]; Btr,Bte=Mb[j[:len(j)//2]],Mb[j[len(j)//2:]]
    G=lambda M:M.T@M/M.shape[0]; q=lambda M,h:float(np.mean((M@h)**2))
    n=Atr.shape[1]
    w,V=eigh(G(Atr)+lam*np.eye(n),G(Btr)+lam*np.eye(n))
    hA=V[:,0]/np.linalg.norm(V[:,0]); hB=V[:,-1]/np.linalg.norm(V[:,-1])
    return q(Bte,hA)/max(q(Ate,hA),1e-300), q(Ate,hB)/max(q(Bte,hB),1e-300)

if __name__=="__main__":
    tok=GPT2Tokenizer.from_pretrained("gpt2"); model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    print("COORDINATE ROBUSTNESS of the reciprocal blind-subspace result (GPT-2)")
    print(f"{'pair':>8} {'sd':>3} " + " ".join(f"{c:>22}" for c in ("u","pc1","diff","pc2d(4-D)")))
    print(f"{'':>8} {'':>3} " + " ".join(f"{'S-blind    J-blind':>22}" for _ in range(4)))
    print("-"*104)
    OUT=[]
    for (lA,lB) in [(5,10),(5,7),(6,9),(7,9)]:
        for sd in (0,1):
            try:
                d=collect_full(model,tok,lA,lB,sd)
                row={}
                for k in ("u","pc1","diff","pc2d"):
                    Zs,Zj=coords(*d,kind=k); row[k]=recip(Zs,Zj)
                OUT.append(dict(pair=[lA,lB],seed=sd,**{k:list(v) for k,v in row.items()}))
                json.dump(OUT,open("coord_robust.json","w"),indent=1)
                print(f"{str((lA,lB)):>8} {sd:>3} " + " ".join(
                    f"{row[k][0]:>10.2g} {row[k][1]:>10.2g}" for k in ("u","pc1","diff","pc2d")))
            except Exception as e:
                print(f"{str((lA,lB)):>8} {sd:>3} ERR {type(e).__name__}: {str(e)[:50]}")
    import numpy as np
    print()
    for k in ("u","pc1","diff","pc2d"):
        a=np.array([r[k] for r in OUT])
        print(f"  {k:>6}: median S-blind {np.median(a[:,0]):>10.3g}  J-blind {np.median(a[:,1]):>9.3g}  "
              f"BOTH>10x {int(np.sum((a[:,0]>10)&(a[:,1]>10)))}/{len(a)}")
