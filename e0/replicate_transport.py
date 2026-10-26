"""PREREGISTERED FRESH-TEMPLATE REPLICATION: collection driver (prereg/PREREG_transport_replication.md, v3 FINAL).

Modes
  --mode pilot   V1: re-collect seed 0 of the ORIGINAL three-template generator for the 12 published and 12 control
                 configurations with the unchanged collect(), and compare with the stored captures (readouts within
                 1e-4 logits; states within 1e-5 relative). Also checks that collect_rerun(bs=16, reverse=False)
                 reproduces collect() exactly, and times the run. Writes pilot_report.json; exit 0 iff V1 passes.
  --mode fresh   Refuses to run unless pilot_report.json records V1 PASS. For each of the 8 frozen templates x 3 frozen
                 prompts x 24 configurations: collect() with the generator's template list replaced by the single frozen
                 template (the prompt is asserted equal to the frozen pool), then the V2 re-execution
                 collect_rerun(bs=7, reverse=True). Every capture is saved (screened or not); every exception and every
                 screen failure is recorded with its reason in manifest.json.

Nothing here analyses the data; analyse_transport_replication.py does, as registered.

    python3 replicate_transport.py --mode pilot
    python3 replicate_transport.py --mode fresh
"""
import argparse, hashlib, json, os, sys, time, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ioi_query_chart as Q                                   # noqa: E402  (collect() is used unchanged)
from transformers import GPT2LMHeadModel, GPT2Tokenizer        # noqa: E402

OUT = "/data/ai_models/explainability/replication_transport"
POOL = next((p for p in (os.path.join(HERE, "..", "prereg", "prompt_pool.json"),          # release tree
                          os.path.join(HERE, "..", "..", "prereg", "prompt_pool.json"))    # repository
             if os.path.exists(p)), None)
SCREEN = 0.15
V1_READOUT_TOL, V1_STATE_REL_TOL = 1e-4, 1e-5
READOUTS, STATES = ("Ys", "Yc", "Yjm", "clean"), ("Zs", "Zc", "Zjm")
ORIGINAL_TEMPLATES = list(Q.TEMPLATES)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def collect_rerun(model, tok, si, nm, seed, templates, n_don=32, bs=16, reverse=False):
    """Copy of ioi_query_chart.collect (circuit='ioi'), differing ONLY in how the intervention rows are batched:
    rows are executed in reversed order when reverse=True, in batches of bs, and returned in the original row order.
    Donor activations and the clean pass are computed exactly as in collect(). With bs=16 and reverse=False it must
    reproduce collect() exactly (checked in the pilot)."""
    rng = np.random.default_rng(seed)
    t = templates[rng.integers(len(templates))]
    a, b = rng.choice(Q.NAMES, 2, replace=False)
    prompt = t.format(A=a, B=b)
    io = tok(" " + a).input_ids[0]; sb = tok(" " + b).input_ids[0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(Q.DEV)
    p = Q.HeadProbe(model, [si, nm])
    vq = Q.query_direction(model, si, nm)
    u = (model.lm_head.weight[io] - model.lm_head.weight[sb]).float()
    dirN = (model.transformer.h[nm[0]].attn.c_proj.weight[nm[1] * Q.HD:(nm[1] + 1) * Q.HD, :].float()
            @ u).cpu().numpy().astype(np.float64)
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
    Sclean, Nclean = torch.tensor(A0[0], device=Q.DEV).float(), torch.tensor(B0[0], device=Q.DEV).float()
    dP = [tok(t.format(A=x, B=y), return_tensors="pt").input_ids[0].tolist()
          for x, y in [rng.choice(Q.NAMES, 2, replace=False) for _ in range(n_don)]]
    dP = [q for q in dP if len(q) == ids.shape[1]]
    dS, dN = [], []
    for i in range(0, len(dP), 8):
        pid = torch.tensor(dP[i:i + 8], device=Q.DEV)
        p.force = {}; model(pid)
        dS.append(p.cache[si].float().clone()); dN.append(p.cache[nm].float().clone())
    dS, dN = torch.cat(dS), torch.cat(dN)

    def run(force_idx, n_rows):
        order = np.arange(n_rows)[::-1] if reverse else np.arange(n_rows)
        A_, B_, Y_ = np.zeros((n_rows, Q.HD)), np.zeros((n_rows, Q.HD)), np.zeros(n_rows)
        for i in range(0, n_rows, bs):
            ix = order[i:i + bs]
            tix = torch.as_tensor(ix.copy(), device=Q.DEV)
            fS, fN = force_idx(tix)
            a_, b_, y_ = fwd(fS, fN, len(ix))
            A_[ix], B_[ix], Y_[ix] = a_, b_, y_
        return A_, B_, Y_

    rows = [(v, x) for v in range(len(dS)) for x in range(Q.N_AL)]
    fA = torch.stack([Sclean + Q.AL[x] * (dS[v] - Sclean) for v, x in rows])
    A1, B1, Y1 = run(lambda ix: (fA[ix], None), len(rows))
    A1c, B1c, Y1c = run(lambda ix: (fA[ix], Nclean.repeat(len(ix), 1)), len(rows))
    fB = torch.stack([Nclean + Q.AL[x] * (dN[v] - Nclean) for v, x in rows])
    A2, B2, Y2 = run(lambda ix: (None, fB[ix]), len(rows))
    pairs = [(i, j) for i in range(0, len(dS), 2) for j in range(0, len(dN), 2)]
    aidx = np.array([((i + j) % Q.N_AL, (3 * i + j) % Q.N_AL) for i, j in pairs])
    fSj = torch.stack([Sclean + Q.AL[k] * (dS[i] - Sclean) for (i, _), k in zip(pairs, aidx[:, 0])])
    fNj = torch.stack([Nclean + Q.AL[k] * (dN[j] - Nclean) for (_, j), k in zip(pairs, aidx[:, 1])])
    A3, B3, Y3 = run(lambda ix: (fSj[ix], fNj[ix]), len(pairs))
    p.close()
    P = lambda A_, B_: np.stack([A_ @ vq, B_ @ dirN], 1)
    return dict(Zs=np.vstack([P(A1, B1), P(A2, B2)]), Ys=np.concatenate([Y1, Y2]),
                Zc=np.vstack([P(A1c, B1c), P(A2, B2)]), Yc=np.concatenate([Y1c, Y2]),
                Zjm=P(A3, B3), Yjm=Y3, Zj=P(A3, B3), Yj=Y3, clean=float(y0[0]), prompt=prompt)


def load_model():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", dtype=torch.float32).to(Q.DEV).eval()
    return model, tok


def configs(pool):
    pub = [(tuple(s), tuple(n), "published") for s in pool["published"]["source"] for n in pool["published"]["target"]]
    ctl = [(tuple(s), tuple(n), "control") for s in pool["controls"]["source"] for n in pool["controls"]["target"]]
    assert [c[:2] for c in pub] == [(s, n) for s in Q.S_INHIB for n in Q.NAME_MOVE], "published heads differ from the snapshot"
    return pub + ctl


def tag(si, nm):
    return f"{si[0]}{si[1]}_{nm[0]}{nm[1]}"


def diffs(new, old):
    out = {}
    for k in READOUTS:
        out[k] = float(np.max(np.abs(np.asarray(new[k], float) - np.asarray(old[k], float))))
    for k in STATES:
        o = np.asarray(old[k], float)
        out[k] = float(np.max(np.abs(np.asarray(new[k], float) - o)) / max(np.max(np.abs(o)), 1e-300))
    return out


def pilot(pool, e0):
    model, tok = load_model()
    Q.TEMPLATES[:] = ORIGINAL_TEMPLATES
    rows, t0 = [], time.time()
    for si, nm, grp in configs(pool):
        stored = os.path.join(e0, f"{'qchart' if grp == 'published' else 'qctrl'}_{tag(si, nm)}_0.npz")
        if not os.path.exists(stored):
            rows.append(dict(config=tag(si, nm), group=grp, stored=None)); continue
        t1 = time.time()
        d = Q.collect(model, tok, si, nm, 0)
        dt = time.time() - t1
        r = dict(config=tag(si, nm), group=grp, stored=os.path.basename(stored), seconds=dt, **diffs(d, np.load(stored)))
        r["pass"] = all(r[k] <= V1_READOUT_TOL for k in READOUTS) and all(r[k] <= V1_STATE_REL_TOL for k in STATES)
        rows.append(r)
        print(f"[V1] {grp:9s} {tag(si, nm):9s} {dt:5.1f}s  max|dY| {max(r[k] for k in READOUTS):.2e}  "
              f"max rel dZ {max(r[k] for k in STATES):.2e}  {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    si, nm, _ = configs(pool)[0]
    d0 = Q.collect(model, tok, si, nm, 0)
    d1 = collect_rerun(model, tok, si, nm, 0, ORIGINAL_TEMPLATES, bs=16, reverse=False)
    eq = {k: float(np.max(np.abs(np.asarray(d0[k], float) - np.asarray(d1[k], float)))) for k in READOUTS + STATES}
    t2 = time.time(); collect_rerun(model, tok, si, nm, 0, ORIGINAL_TEMPLATES, bs=7, reverse=True); rr_s = time.time() - t2
    checked = [r for r in rows if r.get("stored")]
    rep = dict(v1_pass=bool(checked) and all(r["pass"] for r in checked) and len(checked) == 24,
               n_checked=len(checked), tolerances=dict(readout_logits=V1_READOUT_TOL, state_rel=V1_STATE_REL_TOL),
               rerun_equivalence_max_abs=eq, rerun_equivalent=all(v == 0.0 for v in eq.values()),
               seconds_per_capture_median=float(np.median([r["seconds"] for r in checked])) if checked else None,
               seconds_rerun_one=rr_s, total_seconds=time.time() - t0, rows=rows,
               torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               driver_sha256=sha(os.path.abspath(__file__)), pool_sha256=sha(POOL))
    os.makedirs(os.path.join(OUT, "pilot"), exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "pilot", "pilot_report.json"), "w"), indent=1)
    n_est = 576
    print(f"[V1] {'PASS' if rep['v1_pass'] else 'FAIL'}  checked {rep['n_checked']}/24; rerun equivalence "
          f"{'exact' if rep['rerun_equivalent'] else eq}; median {rep['seconds_per_capture_median']:.1f}s/capture + "
          f"{rr_s:.1f}s rerun -> fresh estimate {(n_est * (rep['seconds_per_capture_median'] + rr_s)) / 60:.1f} min")
    return rep["v1_pass"] and rep["rerun_equivalent"]


def fresh(pool):
    rep_path = os.path.join(OUT, "pilot", "pilot_report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    if not (rep.get("v1_pass") and rep.get("rerun_equivalent")):
        raise SystemExit("REFUSING: V1 has not passed (pilot_report.json)")
    if rep.get("driver_sha256") != sha(os.path.abspath(__file__)) or rep.get("pool_sha256") != sha(POOL):
        raise SystemExit("REFUSING: driver or pool changed since the pilot")
    model, tok = load_model()
    outdir = os.path.join(OUT, "fresh"); os.makedirs(outdir, exist_ok=True)
    manifest, t0 = [], time.time()
    for T in pool["admitted"]:
        for pr in T["prompts"]:
            for si, nm, grp in configs(pool):
                name = f"{'pub' if grp == 'published' else 'ctl'}_{tag(si, nm)}_T{T['k']}_{pr['seed']}.npz"
                row = dict(file=name, template=T["k"], seed=pr["seed"], prompt=pr["prompt"], group=grp,
                           source=list(si), target=list(nm))
                try:
                    Q.TEMPLATES[:] = [T["template"]]
                    d = Q.collect(model, tok, si, nm, pr["seed"])
                    assert d["prompt"] == pr["prompt"], (d["prompt"], pr["prompt"])
                    r = collect_rerun(model, tok, si, nm, pr["seed"], [T["template"]], bs=7, reverse=True)
                    assert r["prompt"] == pr["prompt"]
                    sd = float(np.std(d["Yjm"]))
                    row.update(std_Yjm=sd, screened_in=bool(sd >= SCREEN),
                               screen_reason=None if sd >= SCREEN else f"std(Y_joint)={sd:.4f} < {SCREEN}", error=None)
                    np.savez(os.path.join(outdir, name), **{k: v for k, v in d.items() if k != "prompt"},
                             **{"rr_" + k: v for k, v in r.items() if k != "prompt"})
                except Exception as ex:
                    row.update(std_Yjm=None, screened_in=False, screen_reason=None,
                               error=f"{type(ex).__name__}: {ex}", trace=traceback.format_exc()[-800:])
                finally:
                    Q.TEMPLATES[:] = ORIGINAL_TEMPLATES
                manifest.append(row)
                print(f"[FRESH] {len(manifest):3d}/576 {name:32s} "
                      f"{'ERROR ' + row['error'] if row['error'] else ('in ' if row['screened_in'] else 'SCREENED ') + format(row['std_Yjm'], '.3f')}",
                      flush=True)
    json.dump(dict(driver_sha256=sha(os.path.abspath(__file__)), pool_sha256=sha(POOL), seconds=time.time() - t0,
                   rows=manifest), open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    print(f"[FRESH] done: {len(manifest)} captures in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "fresh"])
    ap.add_argument("--e0", default=HERE, help="directory holding the stored seed-0 captures (for V1)")
    A = ap.parse_args()
    torch.set_grad_enabled(False)
    POOL_ = json.load(open(POOL))
    if A.mode == "pilot":
        sys.exit(0 if pilot(POOL_, A.e0) else 3)
    fresh(POOL_)
