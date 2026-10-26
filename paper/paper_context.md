# paper_context — Paper A, arXiv corrected version

Binding contract for the rewrite (plan §F/§G, approved 2026-09-27; replication outcome 2026-09-27).

## Identity
- Title (unchanged): What Can Activation Patching Actually Establish? Reachable-State Geometry and the Limits of Mechanistic Evidence
- Authors (OpenReview record Gfh4EJoWTV): Ankush Kadu, Ananth Kalyanasundaram, Aswanth Krishnan — QpiAI, India.
- Status line: earlier version accepted as a poster at the NeurIPS 2026 Workshop TAE (Trust-AI-Eval); this version corrects
  errors found after acceptance (Appendix A) and adds post-review analyses plus a preregistered replication.
- Venue format: `neurips_2026` `[preprint]`, single column, ~10 main pages, appendix unlimited. No Paper B material.

## One-sentence contribution
Which mechanistic alternatives an activation-patching experiment can rule out is fixed by the internal states the
network reaches under that experiment, so it can be computed before any readout — exactly on operator slices, as
ill-conditioning in trained models — and on a published IOI edge it means single-site evidence cannot identify whether
a head's measured effect transports, which only executed four-point interventions can measure (small, on the tested grids).

## Claims → evidence
| # | claim | level | evidence | status |
|---|---|---|---|---|
| 1 | blind space N(Q,F) and resolving preorder; recompute vs clamp; context vs transport identity | def/identity | §2 | definition |
| 2 | operator on the edge fixes onset d* (2 affine, 5 norm, ∞ softmax/gated) | L1 | Thm 1 (corrected), App B | proved |
| 3 | k-site bound; affine blind fractions 30/50/62.9/71.4% | L1 | Prop 2, App B/C | proved + exact |
| 4 | named generator ≥10× weaker under S than J in 171/185; g_C alias 144/185 (143/28/1/13) | L2 | Table 1 | empirical, re-analysis |
| 5 | two pitfalls: support mismatch (105→9/185), chart selection (103/170→26/120); two claims withdrawn | L2 | §6, App E/F | empirical |
| 6 | IOI: single-site cannot identify transport deviation; executed T_C small (0.0071 logits, 0.43%); replication H1 8/8; no above-control claim | L3 | §7, Table 3, App L/O | exploratory + preregistered replication |

## Terminology (one name per concept)
- reachable set R_Q; blind space N(Q,F); onset d*; Hilbert profile H_Q
- recompute generator g_R=(a−a0)(b−ma−c); clamped (one-at-a-time) generator g_C=(a−a0)(b−b0)
- energy E_Q(h)=hᵀG_Q h; held-out J/S energy ratio; "S-weak" (never "blind" for trained models; never "practically invisible")
- weak identification / ill-conditioned (never "non-identifiable" for trained models)
- capture = (layer pair or head pair, seed); cell = model × task; prompt group = (cell, prompt)
- B-arm swing = ptp of the readout over the single-site B arm (32 donors × 7 α) — the denominator of every "% of swing"
- transport deviation T_C(u,v)=Y(u,v)−Y(u,0)−Y(0,v)+Y(0,0) ("four-point contrast"); never "misattribution", never "increment"
- working-model envelope (identification width of a design under the cubic class); never "bound on the error"
- status tags: [proved] [empirical] [exploratory] [preregistered] [withdrawn]

## Standard T_C wording (every T_C sentence)
"small on the tested grid(s)" + numbers + tag; exploratory data = three base prompts; replication = 24 fresh prompts
from 8 fixed templates, registered smallness criteria met in 8/8 templates; controls at the float32 floor → no
above-control claim. Never: "transport holds", "no interaction", "additive", "above control level" as a finding.

## Outline (topic sentences in main.tex section openers)
1 Intro (Box 1 three levels; contributions; claim-status table) · 2 Setup and estimands (Def 1; Box 2 context vs
transport; conditioning) · 3 Exact theory · 4 Exact → practical (Gram, pencil, protocol box) · 5 Readout-free
prediction (Table 1 with g_C partition) · 6 Two pitfalls · 7 IOI (chart/edge; alias; adequacy table; executed T_C +
replication; reconciliation by scale; non-claim) · 8 Limitations, audit checklist, conclusion.
Appendices: A correction log (generated) · B proofs · C exact numerics · D protocol · E support matching · F chart
selection · G failed hypotheses · H donor/prompt units · I three sites · J robustness · K non-invariant/worst-case ·
L IOI details · M related work · N reproducibility + AI use · O preregistered replication.

## Figure plan
- Fig 1 reachable sets (exact slice; GPT-2 states vs unmatched and matched joint) — claim 1/5; fix labels (C18: 1.64× is one capture)
- Fig 2 regimes (exact residual collapse; practical held-out order statistics — NOT the named generator) — claims 2/4
- Fig 3/4 (appendix) matched spectra, reverse-escape bars + edge scatter — claim 5; relabel per C18
- Fig 5 (new, §7) per-capture scales: single-site envelope, joint envelope, executed RMS T_C; controls grey — claim 6
- Replication panel (optional): template medians vs 0.10-logit and 1% margins
