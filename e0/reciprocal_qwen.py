"""Reciprocal transfer S<->J on Qwen ACTIVE pairs, plus whitened geometry."""
import json, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
from levelC2 import design, kind, DEV
from adjudicate import collect, modfn_qwen
torch.set_grad_enabled(False)
def r2(p,y): return float(1-np.mean((p-y)**2)/max(np.var(y),1e-12))
def erank_w(X):
    Xc=(X-X.mean(0))/(X.std(0)+1e-12); l=np.linalg.svd(Xc,compute_uv=False)**2
    return float(l.sum()**2/(l**2).sum())
M="Qwen/Qwen3.5-9B-Base"
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map=DEV).eval()
names=[n for n in ["Mary","John","Tom","Sarah","Alice","Bob","Emma","David","Laura","Peter","Anna","James","Linda","Paul","Susan","Mark"]
       if len(tok(" "+n,add_special_tokens=False).input_ids)==1]
OUT=[]
print(f"{'pair':>8} {'type':>5} {'sd':>3} {'std Yj':>7} | {'S->S':>7} {'S->J':>11} {'J->J':>7} {'J->S':>8} | {'rk S':>6} {'rk J':>6} {'corr J':>7}")
print("-"*94)
for pr in ["19-23","23-27","27-31","15-23","21-22"]:
    lA,lB=(int(x) for x in pr.split("-"))
    for sd in (0,1,2):
        try:
            Zs,Ys,Zj,Yj,prompt,y0=collect(model,tok,modfn_qwen,lA,lB,sd,names)
            np.savez(f"rec_raw_{lA}_{lB}_{sd}.npz",Zs=Zs,Ys=Ys,Zj=Zj,Yj=Yj)
            mu,s=np.vstack([Zs,Zj]).mean(0),np.vstack([Zs,Zj]).std(0)+1e-12
            W=lambda Z:(Z-mu)/s; Ms,Mj=design(W(Zs),3),design(W(Zj),3)
            rng=np.random.default_rng(sd)
            i=rng.permutation(len(Zs)); a,b=i[:len(i)//2],i[len(i)//2:]
            j=rng.permutation(len(Zj)); c,e=j[:len(j)//2],j[len(j)//2:]
            fs,*_=np.linalg.lstsq(Ms[a],Ys[a],rcond=None); fj,*_=np.linalg.lstsq(Mj[c],Yj[c],rcond=None)
            r=dict(pair=[lA,lB],seed=sd,type=kind(lA)+"->"+kind(lB),std_Yj=float(np.std(Yj)),
                   ss=r2(Ms[b]@fs,Ys[b]),sj=r2(Mj@fs,Yj),jj=r2(Mj[e]@fj,Yj[e]),js=r2(Ms@fj,Ys),
                   rank_s=erank_w(Zs),rank_j=erank_w(Zj),corr_j=float(np.corrcoef(Zj.T)[0,1]))
            OUT.append(r); json.dump(OUT,open("reciprocal_qwen.json","w"),indent=1)
            print(f"{lA:>3}-{lB:<4} {r['type']:>5} {sd:>3} {r['std_Yj']:>7.3f} | {r['ss']:>7.3f} "
                  f"{r['sj']:>11.1f} {r['jj']:>7.3f} {r['js']:>8.2f} | {r['rank_s']:>6.3f} {r['rank_j']:>6.3f} {r['corr_j']:>7.3f}")
        except Exception as ex: print(f"{lA}-{lB} sd{sd} ERR {type(ex).__name__}: {str(ex)[:40]}")
a=np.array([[r['ss'],r['sj'],r['jj'],r['js']] for r in OUT])
print(f"\nMEDIAN  S->S {np.median(a[:,0]):+.3f}  S->J {np.median(a[:,1]):+.1f}  "
      f"J->J {np.median(a[:,2]):+.3f}  J->S {np.median(a[:,3]):+.2f}")
print(f"S->J negative {sum(1 for x in a[:,1] if x<0)}/{len(a)};  J->S negative {sum(1 for x in a[:,3] if x<0)}/{len(a)}")
