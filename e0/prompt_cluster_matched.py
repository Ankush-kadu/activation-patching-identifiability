"""Table 3 (support-matched escapes) with the PROMPT as the unit.

WHY THIS FILE EXISTS (2026-09-23).  Table 3 reports, per model x task cell and protocol, how many captures
escape the extreme-value same-family envelope at each end of the held-out response profile, with a
Holm-adjusted binomial p that treats captures as independent trials.  Captures share prompts: a capture is
a (layer pair, seed) run and the seed selects the prompt, so captures on one prompt share the clean
activations, the donor pool and the task instance.  Two reviews asked for inference at the level of the
independent unit.  Within a cell there are only a few prompts, so per-cell prompt-level tests have almost
no resolution; this file therefore reports, for each protocol and each end, POOLED over cells:

  prompt counts   a prompt (within its cell) escapes under three aggregations of its captures: every
                  capture escapes (strict), a majority escapes, at least one escapes (lenient)
  prompt-level p  one-sided binomial of the strict count against the envelope's nominal 2.5% per-end
                  rate (for the presence of escapes), and of the lenient count against
                  1 - (1 - 0.025)^n_p summed as a Poisson-binomial mean (for their absence, reported as
                  the expected count only)
  cluster bootstrap  resample prompts with replacement (stratified by cell), recompute the pooled
                  capture-level escape fraction, report its 2.5th and 97.5th percentiles

Per-trial escape flags are read from matched_invariant.json (read only); the prompt of each capture from
prompt_cluster.prompt_of_capture().

    python3 prompt_cluster_matched.py      # -> prompt_cluster_matched.json
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from prompt_cluster import prompt_of_capture

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = 0.025
BOOT = 4000
PROTOCOLS = ("frozen", "interior", "symmetric", "matched")
ENDS = {"below": "S-blind end", "above": "reverse (J-blind) end"}


def main():
    os.chdir(HERE)
    M = json.load(open("matched_invariant.json"))["cells"]
    P = prompt_of_capture()
    out = {}
    for proto in PROTOCOLS:
        out[proto] = {}
        for end in ENDS:
            groups = defaultdict(list)                 # (cell, prompt) -> [escape flags]
            cells_n = {}
            for cell, rec in M.items():
                r = rec.get(proto)
                if not r:
                    continue
                cells_n[cell] = r["n"]
                for t in r["per_trial"]:
                    groups[(cell, P.get(t["file"], t["file"]))].append(bool(t[end]))
            keys = sorted(groups)
            strict = sum(all(v) for v in groups.values())
            major = sum(np.mean(v) > 0.5 for v in groups.values())
            lenient = sum(any(v) for v in groups.values())
            caps = [h for v in groups.values() for h in v]
            # cluster bootstrap over prompts, stratified by cell
            by_cell = defaultdict(list)
            for k in keys:
                by_cell[k[0]].append(groups[k])
            rng = np.random.default_rng(0)
            frac = []
            for _ in range(BOOT):
                vals = []
                for c, gl in by_cell.items():
                    for i in rng.integers(0, len(gl), len(gl)):
                        vals.extend(gl[i])
                frac.append(np.mean(vals))
            exp_lenient = float(sum(1 - (1 - NOMINAL) ** len(v) for v in groups.values()))
            out[proto][end] = dict(
                prompts=len(keys), captures=len(caps), capture_escapes=int(sum(caps)),
                prompts_all_escape=int(strict), prompts_majority_escape=int(major), prompts_any_escape=int(lenient),
                p_prompt_all=float(binomtest(strict, len(keys), NOMINAL, alternative="greater").pvalue),
                expected_prompts_any_under_independence=exp_lenient,
                boot_frac_lo=float(np.percentile(frac, 2.5)), boot_frac_hi=float(np.percentile(frac, 97.5)),
                prompts_per_cell={c: len(g) for c, g in by_cell.items()})
            o = out[proto][end]
            print(f"{proto:9s} {ENDS[end]:22s} captures {o['capture_escapes']:>3}/{o['captures']:<3} | prompts all "
                  f"{strict:>2}/{len(keys)}  majority {major:>2}  any {lenient:>2} | p(all) {o['p_prompt_all']:.2e} | "
                  f"cluster-boot fraction [{o['boot_frac_lo']:.3f}, {o['boot_frac_hi']:.3f}]")
    out["estimands"] = dict(
        unit="prompt within a model x task cell; captures on the same prompt are grouped",
        nominal=NOMINAL, boot=f"{BOOT} replicates, prompts resampled with replacement within each cell",
        p_prompt_all="one-sided binomial: prompts on which every capture escapes, against the 2.5% per-end rate")
    json.dump(out, open("prompt_cluster_matched.json", "w"), indent=1)
    print("saved -> prompt_cluster_matched.json")


if __name__ == "__main__":
    main()
