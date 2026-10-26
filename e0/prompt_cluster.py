"""STRESS TEST: the unit is a capture, but captures share PROMPTS.

Table 1 counts captures, and the binomial tests treat them as independent.  They are not fully
independent: a capture is a (layer pair, seed) run, the seed selects the prompt, and across the
matched protocol 185 captures rest on 29 distinct prompts -- so captures within a prompt share the
clean activations, the donor pool and the task instance.  If that correlation is strong the
effective sample size is the number of PROMPTS, not the number of captures, and the reported
p-values are optimistic.

This re-runs the escape analysis with the prompt as the unit:

  per-prompt      a prompt counts as an escape only if EVERY capture on it escapes (the strictest
                  aggregation -- one failure inside a prompt loses the whole prompt)
  cluster bootstrap   resample PROMPTS with replacement, recompute the pooled escape fraction,
                  and report the 2.5th percentile.  This is the standard cluster-robust answer
                  when observations are grouped and the grouping is the thing you doubt.

If the claim survives with prompts as the unit, the capture-level count is a presentation choice
rather than a statistical one, and the paper should say so.
"""
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from donor_split import analyse_donor, envelope
from levelC2 import design
from subspace_test import DEG

NOMINAL = 0.025
BOOT = 2000


def prompt_of_capture():
    """file name -> prompt, from the per-cell matched artifacts."""
    m = {}
    for f in glob.glob("matched_*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("rows", []):
            if "file" in r and "prompt" in r:
                m[r["file"]] = r["prompt"]
    return m


if __name__ == "__main__":
    P = prompt_of_capture()
    cells = defaultdict(list)
    for f in sorted(glob.glob("*_matched_*.npz")):
        task, _, tag, _, _, _ = f[:-4].split("_")
        cells[(task, tag)].append(f)

    per_capture, per_prompt_key = [], []
    rows = {}
    for (task, tag), files in sorted(cells.items()):
        esc, tot, byp = 0, 0, defaultdict(list)
        for f in files:
            r = analyse_donor(*[np.load(f)[k] for k in ("Zs", "Zjm")])
            if r is None:
                continue
            d = np.load(f)
            allZ = np.vstack([d["Zs"], d["Zjm"]])
            mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
            W = lambda Z: (Z - mu) / sd
            cs = np.linalg.norm(design(W(allZ), DEG), axis=0)
            cs[cs == 0] = 1.0
            hit = bool(r["sep_Sblind"] > envelope(design(W(d["Zs"]), DEG) / cs))
            esc += hit
            tot += 1
            key = (task, tag, P.get(f, f))          # prompt identity within the cell
            byp[key].append(hit)
            per_capture.append(hit)
            per_prompt_key.append(key)
        if not tot:
            continue
        # strictest aggregation: a prompt escapes only if all its captures do
        pe = sum(all(v) for v in byp.values())
        rows[f"{tag} {task}"] = dict(captures=tot, cap_escapes=esc,
                                     prompts=len(byp), prompt_escapes=pe)
        print(f"{tag + ' ' + task:>24}  captures {esc:>3}/{tot:<3}   "
              f"prompts {pe:>2}/{len(byp):<2}")

    # ---- cluster bootstrap over prompts, pooled ----
    groups = defaultdict(list)
    for hit, key in zip(per_capture, per_prompt_key):
        groups[key].append(hit)
    keys = list(groups)
    rng = np.random.default_rng(0)
    frac = []
    for _ in range(BOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals = [h for i in pick for h in groups[keys[i]]]
        frac.append(np.mean(vals))
    lo = float(np.percentile(frac, 2.5))

    npr = sum(v["prompts"] for v in rows.values())
    npe = sum(v["prompt_escapes"] for v in rows.values())
    p_prompt = float(binomtest(npe, npr, NOMINAL, alternative="greater").pvalue)
    out = dict(cells=rows, prompts=npr, prompt_escapes=npe, p_prompt=p_prompt,
               captures=len(per_capture), capture_escapes=int(sum(per_capture)),
               boot_lo=lo, boot_reps=BOOT)
    json.dump(out, open("prompt_cluster.json", "w"), indent=1)

    print(f"\n  capture level : {int(sum(per_capture))}/{len(per_capture)}")
    print(f"  PROMPT level  : {npe}/{npr}  (all captures on the prompt must escape), "
          f"binomial p = {p_prompt:.2e}")
    print(f"  cluster bootstrap over prompts: escape fraction 2.5th pct = {lo:.3f} "
          f"(nominal rate {NOMINAL})")
    print("\nsaved -> prompt_cluster.json")
