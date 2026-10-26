"""Single source of truth for model x task cells and raw-capture file naming.

Added 2026-08-27 after the filename-collision bug: nonioi.py saved
{task}_raw_{lA}_{lB}_{seed}.npz with NO model tag, so the Qwen3.5 runs overwrote the GPT-2
captures for the shared layer pair 5-7, and every script that globbed `induction_raw_*` /
`capital_raw_*` pooled both models under the label "GPT-2".  Task captures are now named
{task}_raw_{model_tag}_{lA}_{lB}_{seed}.npz.  The IOI captures keep their historical prefixes
(adj_raw_* = GPT-2 IOI, rec_raw_* = Qwen3.5-9B-Base IOI, active pairs).
"""
MODEL_TAGS = {"gpt2": "gpt2",
              "Qwen/Qwen3.5-9B-Base": "qwen35", "Qwen3.5-9B-Base": "qwen35",
              "Qwen/Qwen3-8B": "qwen3", "Qwen3-8B": "qwen3"}
MODEL_LABELS = {"gpt2": "GPT-2", "qwen35": "Qwen3.5", "qwen3": "Qwen3-8B"}


def model_tag(name):
    return MODEL_TAGS.get(name, name.split("/")[-1].lower().replace("-", "").replace(".", ""))


def raw_name(task, tag, lA, lB, seed):
    return f"{task}_raw_{tag}_{lA}_{lB}_{seed}.npz"


# One entry per model x task cell.
#   pattern     glob for the raw captures of that cell (Zs, Ys, Zj, Yj)
#   label       human label used in JSON artifacts and figures
#   pre         LaTeX macro prefix (letters only) used by ledger_paperA.py
#   nonioi      (task, model-name-in-json) for the nonioi_{task}_{model}.json artifact, or None
CELLS = [
    dict(pattern="adj_raw_*.npz",              label="GPT-2 IOI",         pre="GPT",     nonioi=None),
    dict(pattern="rec_raw_*.npz",              label="Qwen3.5 IOI",       pre="Qwen",    nonioi=None),
    dict(pattern="induction_raw_gpt2_*.npz",   label="GPT-2 induction",   pre="IndGPT",  nonioi=("induction", "gpt2")),
    dict(pattern="induction_raw_qwen35_*.npz", label="Qwen3.5 induction", pre="IndQwen", nonioi=("induction", "Qwen3.5-9B-Base")),
    dict(pattern="capital_raw_gpt2_*.npz",     label="GPT-2 capital",     pre="CapGPT",  nonioi=("capital", "gpt2")),
    dict(pattern="capital_raw_qwen35_*.npz",   label="Qwen3.5 capital",   pre="CapQwen", nonioi=("capital", "Qwen3.5-9B-Base")),
    dict(pattern="greater_raw_gpt2_*.npz",     label="GPT-2 greater",     pre="GtGPT",   nonioi=("greater", "gpt2")),
    dict(pattern="agreement_raw_gpt2_*.npz",   label="GPT-2 agreement",   pre="AgrGPT",  nonioi=("agreement", "gpt2")),
]

BY_LABEL = {c["label"]: c for c in CELLS}

# ---------------------------------------------------------------- alpha grid ---
# Single-site captures are stored as [patch-A arm ; patch-B arm], each arm donor-major with
# N_AL interpolation coefficients per donor:  row = donor_index * N_AL + alpha_index.
N_AL = 7
ALPHAS = [-0.5, -1 / 6, 1 / 6, 0.5, 5 / 6, 7 / 6, 1.5]        # np.linspace(-0.5, 1.5, 7)
INTERIOR = [i for i, a in enumerate(ALPHAS) if 0.0 <= a <= 1.0]   # -> [2, 3, 4]


def restrict_alpha(Zs, keep=INTERIOR, n_al=N_AL):
    """Keep only the given alpha indices, in both arms.

    The joint family is built from RAW donors (alpha = 1), while the single-site family sweeps
    alpha in [-0.5, 1.5] and so reaches states ~1.9x outside the joint family's box.  A
    direction that is large only in that extrapolated region would look 'visible to S, blind to
    J' for a reason that has nothing to do with the mechanism.  Restricting to the interior
    alphas is the conservative check; the definitive fix is to give J the same alpha grid.
    """
    import numpy as _np
    n = len(Zs) // 2
    idx = _np.array([r for r in range(n) if r % n_al in keep])
    return _np.vstack([Zs[idx], Zs[n + idx]])
