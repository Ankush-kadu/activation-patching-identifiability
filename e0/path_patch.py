"""MUST-DO: does WANG-STYLE PATH PATCHING (S-Inhib -> Name Mover QUERY) itself
have a blind subspace, or is that richer evidence family sufficient?

Wang et al. established S-Inhibition -> Name Mover by path patching specifically
into the Name Movers' QUERY vectors. That is a STRICTLY RICHER evidence family
than the component-wise patching tested so far. If it also has a blind subspace,
the result generalizes to the actual evidence used. If it does NOT, our framework
explains WHY that richer evidence sufficed -- which is equally publishable.

Path intervention: replace head A's contribution to the residual ONLY along the
route into head B's QUERY. K and V for head B, and every other consumer of the
residual, keep their clean values.
"""
import json, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
DEV="cuda:0" if torch.cuda.is_available() else "cpu"
NH,HD,D=12,64,768
S_INHIB=[(7,3),(7,9),(8,6),(8,10)]; NAME_MOVE=[(9,9),(9,6),(10,0)]
TEMPLATES=["When {A} and {B} went to the store, {B} gave a drink to",
           "When {A} and {B} went to the park, {B} handed a ball to",
           "After {A} and {B} finished work, {B} gave the keys to"]
NAMES=["Mary","John","Tom","Sarah","Alice","Bob","Emma","David","Laura","Peter","Anna","James"]

class PathProbe:
    """Patch head A's residual contribution ONLY into head B's query path."""
    def __init__(s, model, lA, hA, lB, hB):
        s.m=model; s.lA,s.hA,s.lB,s.hB=lA,hA,lB,hB
        s.zA=None; s.delta=None; s.resid=None; s.cache={}; s._busy=False
        s._h=[]
        s._h.append(model.transformer.h[lA].attn.c_proj.register_forward_pre_hook(s._readA()))
        s._h.append(model.transformer.h[lB].attn.c_proj.register_forward_pre_hook(s._readB()))
        s._h.append(model.transformer.h[lB].ln_1.register_forward_pre_hook(s._readR()))
        s._h.append(model.transformer.h[lB].attn.c_attn.register_forward_hook(s._qpatch()))
    def _readA(s):
        sl=slice(s.hA*HD,(s.hA+1)*HD)
        def f(mod,args):
            s.cache["zA"]=args[0][:,-1,sl].detach().clone(); return None
        return f
    def _readB(s):
        sl=slice(s.hB*HD,(s.hB+1)*HD)
        def f(mod,args):
            if not s._busy: s.cache["zB"]=args[0][:,-1,sl].detach().clone()
            return None
        return f
    def _readR(s):
        def f(mod,args):
            if not s._busy: s.resid=args[0].detach().clone()
            return None
        return f
    def _qpatch(s):
        def f(mod,args,out):
            if s.delta is None or s._busy: return out
            s._busy=True
            try:
                ln=s.m.transformer.h[s.lB].ln_1
                r=s.resid.clone(); r[:,-1,:]=r[:,-1,:]+s.delta.to(r.dtype)
                # layernorm computed manually (avoid re-entering the ln_1 pre-hook)
                mu=r.mean(-1,keepdim=True); var=r.var(-1,keepdim=True,unbiased=False)
                xn=(r-mu)/torch.sqrt(var+ln.eps)*ln.weight+ln.bias
                # Conv1D computed manually (avoid re-entering THIS hook)
                out_mod=xn@mod.weight+mod.bias
                o=out.clone(); o[:,-1,:D]=out_mod[:,-1,:D]   # Q only; K,V stay clean
            finally:
                s._busy=False
            return o
        return f
    def close(s):
        for h in s._h: h.remove()

def contrib(model,l,h,z):
    W=model.transformer.h[l].attn.c_proj.weight[h*HD:(h+1)*HD,:].float()
    return z.float()@W

def run_pair(model,tok,lA,hA,lB,hB,seed=0,n_don=24,n_al=7):
    rng=np.random.default_rng(seed); t=TEMPLATES[rng.integers(len(TEMPLATES))]
    a,b=rng.choice(NAMES,2,replace=False); prompt=t.format(A=a,B=b)
    io=tok(" "+a).input_ids[0]; sb=tok(" "+b).input_ids[0]
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    u=(model.lm_head.weight[io]-model.lm_head.weight[sb]).float()
    p=PathProbe(model,lA,hA,lB,hB)
    p.delta=None; lg=model(ids).logits[0,-1].float()
    zA_clean=p.cache["zA"][0].clone(); zB_clean=p.cache["zB"][0].clone()
    y0=float(lg[io]-lg[sb])
    # donors
    dz=[]
    dP=[t.format(A=x,B=y) for x,y in [rng.choice(NAMES,2,replace=False) for _ in range(n_don)]]
    for q in dP:
        p.delta=None; model(tok(q,return_tensors="pt").input_ids.to(DEV))
        dz.append(p.cache["zA"][0].clone())
    dz=torch.stack(dz)
    al=np.linspace(-0.5,1.5,n_al)
    A_,B_,Y_=[],[],[]
    for v in dz:
        for x in al:
            z=zA_clean+x*(v-zA_clean)
            p.delta=contrib(model,lA,hA,z)-contrib(model,lA,hA,zA_clean)
            lg=model(ids).logits[0,-1].float(); p.delta=None
            A_.append(float(contrib(model,lA,hA,z)@u))
            B_.append(float(contrib(model,lB,hB,p.cache["zB"][0])@u))
            Y_.append(float(lg[io]-lg[sb]))
    # ---- matched JOINT family: same prompt, same heads, independent A and B ----
    dzB=[]
    for qq in dP:
        p.delta=None; model(tok(qq,return_tensors="pt").input_ids.to(DEV))
        dzB.append(p.cache["zB"][0].clone())
    dzB=torch.stack(dzB)
    Aj,Bj,Yj=[],[],[]
    sl=slice(hB*HD,(hB+1)*HD)
    def forceB(zb):
        def f(mod,args):
            x=args[0].clone(); x[:,-1,sl]=zb.to(x.dtype); return (x,)+tuple(args[1:])
        return f
    for i in range(0,len(dz),2):
        for j in range(0,len(dzB),2):
            p.delta=contrib(model,lA,hA,dz[i])-contrib(model,lA,hA,zA_clean)
            hb=model.transformer.h[lB].attn.c_proj.register_forward_pre_hook(forceB(dzB[j]))
            lg=model(ids).logits[0,-1].float(); hb.remove(); p.delta=None
            Aj.append(float(contrib(model,lA,hA,dz[i])@u))
            Bj.append(float(contrib(model,lB,hB,dzB[j])@u))
            Yj.append(float(lg[io]-lg[sb]))
    p.close()
    return (np.stack([np.array(A_),np.array(B_)],1),np.array(Y_),
            np.stack([np.array(Aj),np.array(Bj)],1),np.array(Yj),prompt,y0)

if __name__=="__main__":
    tok=GPT2Tokenizer.from_pretrained("gpt2"); model=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    print("WANG-STYLE PATH PATCHING: S-Inhibition -> Name Mover QUERY")
    print("Does restricting the intervention to the published ROUTE change what is identifiable?\n")
    print(f"{'S-Inhib':>9} {'NameMover':>10} {'sd':>3} {'clean':>7} {'std(Y)':>8} {'range a':>10}  effect of Q-path patch")
    print("-"*80)
    OUT=[]
    for (lA,hA) in [(8,6),(7,9),(8,10)]:
        for (lB,hB) in [(9,9)]:
            for sd in (0,1,2):
                try:
                    Z,Y,Zj,Yj,prompt,y0=run_pair(model,tok,lA,hA,lB,hB,sd)
                    np.savez(f"path_raw_{lA}{hA}_{lB}{hB}_{sd}.npz",Z=Z,Y=Y,Zj=Zj,Yj=Yj)
                    A=Z[:,0]
                    r=dict(sA=[lA,hA],nm=[lB,hB],seed=sd,clean=y0,stdY=float(np.std(Y)),
                           rangeA=float(np.ptp(A)),rangeB=float(np.ptp(Z[:,1])),
                           rangeY=float(np.ptp(Y)),corr=float(np.corrcoef(A,Y)[0,1]))
                    OUT.append(r); json.dump(OUT,open("path_patch.json","w"),indent=1)
                    print(f"{str((lA,hA)):>9} {str((lB,hB)):>10} {sd:>3} {y0:>7.3f} {r['stdY']:>8.4f} "
                          f"{r['rangeA']:>10.3f}  dY range {r['rangeY']:.4f}, corr(a,Y)={r['corr']:+.3f}")
                except Exception as e:
                    print(f"{str((lA,hA)):>9} ERR {type(e).__name__}: {str(e)[:60]}")
    if not OUT: raise SystemExit('no successful trials')
    a=np.array([[r['stdY'],r['rangeY'],r['corr']] for r in OUT])
    print(f"\nMEDIAN std(Y) under Q-path patching = {np.median(a[:,0]):.4f}   "
          f"median |dY| range = {np.median(a[:,1]):.4f}   median corr = {np.median(a[:,2]):+.3f}")
    print("\nIf the Q-path family moves Y substantially, it is a genuinely informative")
    print("evidence family and the next step is to measure ITS blind subspace.")
