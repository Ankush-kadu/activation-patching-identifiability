# Preregistration v3 (FINAL): fresh-template replication of the executed four-point transport contrast

**Status.** FINAL, approved by the author on 2026-09-27 with four last amendments (listed at the end). Frozen together
with the implementation scripts by commit and tag `prereg-transport-v3-frozen`, with sha256 values in
`prereg/FREEZE.sha256`, BEFORE the V1 pilot and before any model forward pass of this study. If V1 passes, the fresh
replication proceeds with no further scientific changes.

The implementation is the collection driver `repro/e0/replicate_transport.py` and the analysis script
`repro/e0/analyse_transport_replication.py`; both are frozen with this file.

Version 1 (3 templates × 8 prompts) is superseded by the author's seven amendments of 2026-09-27.

## 1. Question and status of the prior evidence

On the published GPT-2 small IOI head pairs (S-Inhibition → Name Mover), the executed four-point transport deviation
is

T_C(u,v) = Y(u,v) − Y(u,0) − Y(0,v) + Y(0,0).

It was small in an exploratory, post-review analysis of three previously inspected base prompts from three templates:
- median RMS 0.00709 logits (`AdqRmsT`);
- 0.427% of the B-arm swing (`AdqRmsTPct`);
- above the control-head level in 7/7 paired (Name Mover, prompt) cells, with ratios 12.7–50.2 (`AdqPairedCells`,
  `AdqPairedRatioMin`/`Max`).

This study asks whether that holds on fresh prompts from templates this paper's analyses have never used.

## 2. Templates and prompts: the frozen pool

The pool is `prereg/prompt_pool.json` (sha256 `fddb3400a986fbca3dd49cca3b62efc3b8b4815b379acea1f80497d6e9cc91b7`),
produced deterministically by `prereg/build_prompt_pool.py`. The build uses the GPT-2 tokeniser only; no model forward
pass was run.

- **Task family.** The IOI templates of Wang et al. (2023), copied verbatim from the authors' public code
  (`redwoodresearch/Easy-Transformer` @ `ea15315`, `easy_transformer/ioi_dataset.py`, `BABA_TEMPLATES`, 15 templates).
  - Each template is converted to the ABBA order of the existing generator: IO = the first-mentioned name A; subject
    S = B, repeated. The conversion is the same first-clause swap the source code applies.
  - Each template is truncated before the final IO name.
- **Previously inspected templates.** This paper's analyses used exactly four templates:
  - `ioi_query_chart.py` (query-chart captures): "When {A} and {B} went to the store, {B} gave a drink to" / "… went
    to the park, {B} handed a ball to" / "After {A} and {B} finished work, {B} gave the keys to";
  - `levelC2.py` (matched captures), which also adds "Because {A} and {B} were hungry, {B} passed the bread to".
- **Admission rule (genuine independence).** A source template is admitted only if its first clause (the joint event,
  ignoring the opening connective, punctuation and PLACE/OBJECT fillers) differs from every inspected template and from
  every other admitted template. Where several source templates share a first clause, the first-listed one is kept.
  - This admits **8 templates**: Wang list positions 2, 3, 4, 5, 7, 10, 14 and 15.
  - It excludes 7:
    - positions 1, 6, 11 and 12 share the inspected frame "went to the PLACE";
    - positions 8, 9 and 13 duplicate an admitted first clause.
  - The source family has no further distinct frames: the "long", "late-IO" and "early-IO" lists vary only prefixes
    and connectives of the same 15 frames.
  - Twelve independent templates are therefore not available. The design is **8 templates × 3 prompts** (N = 24).
- **Fillers.** PLACE and OBJECT fillers are assigned in template order from the source lists, after removing words
  used by the inspected templates ("store", "drink").
- **Admitted templates** (token lengths are constant over all name pairs):

  | T | Template (prompt ends before the IO name) | Source position |
  |---|---|---|
  | T1 | Then, {A} and {B} had a lot of fun at the garden. {B} gave a ring to | 2 |
  | T2 | Then, {A} and {B} were working at the restaurant. {B} decided to give a kiss to | 3 |
  | T3 | Then, {A} and {B} were thinking about going to the school. {B} wanted to give a bone to | 4 |
  | T4 | Then, {A} and {B} had a long argument, and afterwards {B} said to | 5 |
  | T5 | When {A} and {B} got a basketball at the hospital, {B} decided to give it to | 7 |
  | T6 | While {A} and {B} were commuting to the office, {B} gave a computer to | 10 |
  | T7 | The house {A} and {B} went to had a necklace. {B} gave it to | 14 (flagged: lexical overlap "went to"; sensitivity analysis S1) |
  | T8 | Friends {A} and {B} found a snack at the station. {B} gave it to | 15 |

- **Prompts.**
  - The unchanged generator (`ioi_query_chart.collect`, snapshot copy) is called with its template list replaced by
    the single template T_k.
  - For seed s it draws `rng = default_rng(s)`; `rng.integers(1)`; (A, B) = `rng.choice(NAMES, 2, replace=False)`
    from the generator's 12 names; then the 32 donor name pairs.
  - For T_k, seeds 5000 + 100k + j (j = 0, 1, …) are scanned. The first three with distinct (A, B) are admitted.
  - The 24 admitted prompts, their seeds and all donor pairs are listed in `prompt_pool.json`.
  - No admitted prompt equals an inspected prompt; the admitted templates differ from the inspected ones.
- **Prior use.** The Wang template family underlies public IOI benchmark data that the authors' separate
  circuit-selection work used. None of this paper's analyses (query-chart captures, matched captures, the four-point
  contrast) used these templates. A scan of the repository's executed sources finds none of their first clauses.
  They are uninspected for this estimand and these interventions.
- **Scope of inference.** The eight templates are treated as the complete, FIXED evaluation set, not as an i.i.d.
  sample from a population of templates. The result covers these eight fixed frames and the 24 preregistered
  prompts only: GPT-2 small, the fixed fillers and the published head pairs. It does not imply generalisation over
  all name pairs from the 12-name pool, over other templates, models or heads, and no population-level template
  p-value is claimed.

## 3. Head configurations (fixed before any T_C analysis)

- **Published** (12 per prompt): S-Inhibition {L7H3, L7H9, L8H6, L8H10} × Name Mover {L9H9, L9H6, L10H0}.
- **Controls** (12 per prompt): early heads {L0H5, L1H2, L2H9, L3H4} × the same Name Movers {L9H9, L9H6, L10H0}.
  - These are the snapshot script's `CTRL_A` × `CTRL_B`.
  - **Pairing:** within a prompt, the four published configurations and the four control configurations that share a
    Name Mover form one paired cell. There are 3 cells per prompt.
  - **When the controls were chosen.** They were specified on 2026-08-28 (commit `aba38d7`), together with the
    exploratory control captures, for the single-site detection analysis. That was 27 days before the four-point
    transport contrast was first computed (2026-09-24, commit `941e5ea`). No control was chosen, added or removed
    after inspecting T_C.
  - The control heads' exploratory T_C values (median 1.88×10⁻⁴ logits) were seen before this preregistration. That
    is a disclosure, not a selection.
- **Inclusion screen** (unchanged): a configuration is analysed only if std(Y_joint) ≥ 0.15. Every configuration
  that fails the screen or raises an error is recorded with its reason (the exploratory collector did not do this;
  correction C21).
- **Captures:** 24 prompts × 24 configurations = 576.
- **Collection design** (unchanged):
  - GPT-2 small, float32;
  - pre-W_O head outputs forced at the final position;
  - 32 donors; α ∈ {−0.5, −1/6, 1/6, 0.5, 5/6, 7/6, 1.5};
  - single-site arms (patch-A with recompute; patch-B with A pinned);
  - clamped arm;
  - 16 × 16 joint grid with the donor-id dose rule of the query-chart collector.

## 4. Estimands and the inferential unit (prompts nested within templates)

Per capture c:
- RMS_T(c) = sqrt(mean over the 256 joint states of T_C²), in logits;
- rel(c) = 100 · RMS_T(c) / swing(c), where swing = max − min of the B-arm single-site readouts.

Per prompt p:
- M_p = median of RMS_T over the screened published configurations;
- M^rel_p = median of rel over the same configurations.

Per template t (the confirmatory unit):
- V_t = median of M_p over the template's analysable prompts;
- V^rel_t likewise.

Central values: the median of V_t over templates and the median of V^rel_t over templates.

A prompt is analysable if at least one published configuration passes the screen. A template is analysable if at
least one of its prompts is. Excluded prompts and templates are reported.

## 5. Hypotheses

Finite-set criteria over the eight fixed templates (§4 gives the template values):

**H1_abs (small, absolute).** Passes iff at least 7 of the 8 template values V_t are < 0.10 logits.

**H1_rel (small, relative).** Passes iff at least 7 of the 8 template values V^rel_t are < 1% of the B-arm swing.

**H2 (above the control level).** Passes iff the published RMS T_C exceeds the control RMS T_C in at least 7 of the 8
templates (d_t > 0), AND the median over the 8 templates of d_t is positive, AND both the published and the control
deviations are numerically resolved under V2 (§7). No materiality multiple is attached: the claim is "above the control
level", not "substantially above".

A template that is not analysable counts as NOT meeting the criterion concerned.

**The margins (0.10 logits; 1% of the swing) were selected after the exploratory three-prompt analysis and before any
fresh-template data exist.** Both absolute and relative estimates, with their bounds, will be reported in the paper
whatever the outcome.

## 6. Confirmatory analysis (finite-set criteria over the eight fixed templates)

- **H1_abs, H1_rel:** count the templates with V_t < 0.10 logits and with V^rel_t < 1%. Each criterion passes at
  7 or more of 8.
- **H2:**
  - For paired cell (p, n): d_{p,n} = log(median published RMS_T with Name Mover n) − log(median control RMS_T with
    Name Mover n). Defined when both sides have ≥ 1 screened configuration.
  - d_p = median over the prompt's defined cells. d_t = median of d_p over the template's prompts.
  - Count the templates with d_t > 0 and compute the median of d_t over the 8 templates.
  - Apply the §7 floor rules.
- **Reported regardless of outcome:**
  - all eight values of V_t, V^rel_t and d_t;
  - the medians over templates;
  - the ratio exp(median d_t).
  These are descriptive summaries of the fixed set; no confidence intervals or p-values over templates are computed.
- **Headline statement** ("small in absolute and relative terms, and above the control level"): it requires H1_abs,
  H1_rel and H2 to pass together, and all validity gates to pass.

## 7. Validity gates (checked in this order; a failure stops the run and is reported as registered)

- **V1 reproduction (pilot).**
  - Re-collect seed 0 of the ORIGINAL three-template generator (the already inspected published and control
    configurations) with the replication driver.
  - Compare with the stored captures: the maximum stored-versus-recollected readout difference must be
    ≤ 1×10⁻⁴ logits, and every state coordinate within 1e-5 relative (max |difference| / max |stored| per array).
  - This tolerance will not be relaxed after the pilot is seen.
  - The pilot also checks that the V2 re-execution function reproduces the driver's collection exactly when run with
    the original batching, and it times the run.
  - If V1 fails, the study stops and the failure is reported.
- **V2 numerical floor.** V2 measures batch-composition numerical effects separately from V1.
  - For every fresh capture, re-execute every intervention row in reversed row order with batch size 7 instead of 16.
    This covers the single-site, clamped and joint rows, a superset of the clamped and joint rows named in v2. Donor
    activations and the clean pass are unchanged.
  - Recompute T_C. The floor is the RMS of (re-executed − original T_C) over the 256 joint states.
  - A capture's deviation is **resolved** if its RMS_T exceeds 10× its floor. A family (published or control) is
    resolved if ≥ 90% of its analysed captures are resolved.
  - H2 can support "above control" only if BOTH families are resolved.
  - If the published family is resolved but the control family is not, the exact outcome is reported: published
    deviations resolved, control deviations at the numerical floor. No calibrated published/control ratio and no
    formal above-control result is claimed.
  - If the published family itself is not resolved, H2 fails.
  - The floors of both families are reported regardless.
- **V3 identity.** The injected vectors reproduce the recorded corner coordinates:
  - joint a = clamped-arm a;
  - joint b = B-arm b;
  - pinned-arm a spread = 0;
  - clamped-arm b spread ≤ 1e-12.
- **V4 screen.** At least 50% of published configurations and at least 5 templates remain analysable. Note that
  under the 7-of-8 criteria, two non-analysable templates already make H1 and H2 fail.
- **V5 swing.** Swing ≥ 0.5 logits in every analysed capture; otherwise the capture is excluded and reported.

## 8. Sample size

N = 24 prompts (8 templates × 3 prompts) is fixed prospectively as a precision and replication sample. It is set by
the number of genuinely distinct, uninspected templates available (8) and by compute. It is not formally powered: the
three exploratory prompts cannot support a power calculation. N cannot change after fresh data are seen.

**Planning sensitivity (not a power claim).** The exploratory per-prompt medians are 0.0046–0.011 logits; the
between-prompt SD of the log median is 0.44 and the within-prompt SD 0.68. Under that dispersion:
- H1_abs fails only if at least two of the eight template medians reach 0.10 logits, about 10× the exploratory values.
- The exploratory control deviations (about 2×10⁻⁴ logits) may lie near the float32 batch-composition floor. In that
  case the floor rule of §7 decides H2's reportable outcome, whatever the sign of d_t.

## 9. Pre-specified secondary and sensitivity analyses (descriptive; they cannot change any decision)

- S1: all confirmatory statistics recomputed excluding T7.
- S2: a prompt-level analysis with the 24 prompts as units, labelled descriptive because prompts are nested in
  templates.
- S3: a no-screen analysis.
- S4: a per-template table of V_t, V^rel_t and d_t.
- S5: the median |T_C|.
- S6: the single-site cubic's held-out prediction error for T_C against the zero predictor (donor partition as in the
  exploratory analysis).
- S7: the single-site and joint working-model envelopes and their coverage of the executed T_C.
- S8: single-site Wald rejections under pooled and robust covariances, and c_S versus c_J.
- S9: donor reuse — per prompt, how many donor name pairs recur in other prompts of the same template.

## 10. Decision rule for the paper

- **Upgrade to a preregistered result.** §7 of the paper states the transport result as a preregistered
  fresh-template finding ONLY IF V1–V5 all pass AND H1_abs, H1_rel and H2 all pass under the finite-set criteria. The
  wording stays scoped to the eight fixed frames, the 24 preregistered prompts and the tested grid, and it reports
  absolute and relative estimates and all eight template values.
- **Otherwise the registered outcome is reported exactly:**
  - H1_abs or H1_rel fails: the corresponding smallness claim did not replicate on the fixed fresh set.
  - Controls floor-limited: "published deviations resolved; control deviations at the numerical floor", with no
    ratio or above-control claim.
  - Published deviations unresolved: H2 fails.
  - H2 fails on its counts: "above the control level" is not claimed.
  - Any other validity failure: the exploratory wording is kept and the failure is reported.
- **After fresh collection begins, nothing changes:** no optional stopping, reseeding, template or prompt changes,
  control changes, margin changes, or analysis changes. Any further study needs a new preregistration.

## 11. Compute estimate (analytic; the V1 pilot confirms it before fresh collection)

- Per capture: about 960 GPT-2-small forward passes on 15–20-token prompts (448 single-site + 224 clamped + 256 joint
  + 32 donor + clean), plus about 480 for the V2 re-execution; batch size 16.
- 576 captures come to about 0.83 M passes, or about 52 k batches.
- At about 5–15 ms per batch of 16 on an A100 with hooks, that is **about 5–15 minutes of GPU time** plus model load.
- Outputs: about 576 × 35 kB ≈ 20 MB under `/data/ai_models/explainability/replication_transport/`.

## 12. Frozen files

- This file, `prereg/prompt_pool.json`, `prereg/build_prompt_pool.py`, `repro/e0/replicate_transport.py` and
  `repro/e0/analyse_transport_replication.py`.
- Their sha256 values are in `prereg/FREEZE.sha256`, committed and tagged `prereg-transport-v3-frozen` before V1.

## Amendments in v3 (author, 2026-09-27; before any model forward pass)

1. The eight templates are a fixed evaluation set. The confidence-bound and Wilcoxon inference is replaced by the
   finite-set criteria: 7 of 8 below 0.10 logits; 7 of 8 below 1%; 7 of 8 with d_t > 0 plus a positive median d_t.
   No population-level template p-values.
2. H2 requires both families to be numerically resolved under V2. When controls are floor-limited, that exact
   outcome is reported without a ratio or an above-control claim. When published deviations are unresolved, H2 fails.
3. The V1 readout tolerance is 1×10⁻⁴ logits and will not be relaxed after the pilot. V2 measures batch-composition
   effects separately.
4. The scope is the eight fixed frames and the 24 preregistered prompts, with no implied generalisation over all name
   pairs of the 12-name pool.

Unchanged: N = 24, the 0.10-logit and 1%-of-swing margins, the published and control head sets, and every other
validity gate.
