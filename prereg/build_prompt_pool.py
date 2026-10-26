"""Freeze the fresh-prompt pool for the transport replication (CPU only; tokeniser, no model forward pass).

Templates: the IOI family of Wang et al. (2023), taken verbatim from the authors' public code
(redwoodresearch/Easy-Transformer, easy_transformer/ioi_dataset.py, BABA_TEMPLATES), converted to the ABBA order that
the existing generator uses (IO = first-mentioned name A, subject S = B repeated) by the same first-clause swap the
source code applies, truncated before the final IO name. A template is ADMITTED only if its first clause (the joint
event, ignoring the opening connective, punctuation and the PLACE/OBJECT fillers) differs from every template this
paper's analyses have used and from every other admitted template; within a set of templates sharing a first clause the
first-listed one is kept. PLACE/OBJECT fillers are assigned in template order from the source lists after removing
words used by the inspected templates.

Prompts: the generator (ioi_query_chart.collect, unchanged) is called with its template list replaced by the single
admitted template, so for seed s it draws rng = default_rng(s); rng.integers(1); (A, B) = rng.choice(NAMES, 2,
replace=False); then 32 donor name pairs rng.choice(NAMES, 2, replace=False). For admitted template k (1..8) seeds
5000 + 100 k + j, j = 0, 1, ... are scanned and the first three with distinct (A, B) are admitted. This replays
exactly those draws and records prompts and donor pairs.

    python3 build_prompt_pool.py      # -> prompt_pool.json (deterministic)
"""
import hashlib, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Mary", "John", "Tom", "Sarah", "Alice", "Bob", "Emma", "David", "Laura", "Peter", "Anna", "James"]
INSPECTED = {  # every IOI template used by this paper's analyses (ioi_query_chart.TEMPLATES, levelC2.TEMPLATES)
    "When {A} and {B} went to the store, {B} gave a drink to": "ioi_query_chart.py, levelC2.py",
    "When {A} and {B} went to the park, {B} handed a ball to": "ioi_query_chart.py, levelC2.py",
    "After {A} and {B} finished work, {B} gave the keys to": "ioi_query_chart.py, levelC2.py",
    "Because {A} and {B} were hungry, {B} passed the bread to": "levelC2.py",
}
SOURCE = dict(repo="https://github.com/redwoodresearch/Easy-Transformer", commit="ea15315",
              file="easy_transformer/ioi_dataset.py", list="BABA_TEMPLATES")
BABA = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]
# first clause (joint event) of each source template, as judged by the stated rule; None-free and explicit
FIRST_CLAUSE = ["went to the PLACE", "had a lot of fun at the PLACE", "were working at the PLACE",
                "were thinking about going to the PLACE", "had a long argument", "went to the PLACE",
                "got a OBJECT at the PLACE", "got a OBJECT at the PLACE", "were working at the PLACE",
                "were commuting to the PLACE", "went to the PLACE", "went to the PLACE", "had a long argument",
                "PLACE ... went to had a OBJECT", "found a OBJECT at the PLACE"]
INSPECTED_FIRST_CLAUSES = {"went to the PLACE", "finished work", "were hungry"}
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]
INSPECTED_WORDS = {"store", "drink", "park", "ball", "work", "keys", "bread"}
PUBLISHED = dict(source=[(7, 3), (7, 9), (8, 6), (8, 10)], target=[(9, 9), (9, 6), (10, 0)])
CONTROLS = dict(source=[(0, 5), (1, 2), (2, 9), (3, 4)], target=[(9, 9), (9, 6), (10, 0)])


def abba(t):
    """Wang et al.'s own conversion: swap [B]/[A] in the first clause only; then drop the final ' [A]'."""
    s, first = t, True
    out = list(s)
    i = 1
    while i < len(s) - 1:
        if s[i - 1:i + 2] == "[B]" and first:
            out[i] = "A"
        elif s[i - 1:i + 2] == "[A]" and first:
            first = False; out[i] = "B"
        i += 1
    s = "".join(out)
    assert s.endswith(" [A]"), s
    s = s[:-len(" [A]")]
    return s.replace("[A]", "{A}").replace("[B]", "{B}")


def main():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    places = [p for p in PLACES if p not in INSPECTED_WORDS]
    objects = [o for o in OBJECTS if o not in INSPECTED_WORDS]
    admitted, excluded, seen = [], [], set()
    ip = io = 0
    for idx, (t, fc) in enumerate(zip(BABA, FIRST_CLAUSE)):
        if fc in INSPECTED_FIRST_CLAUSES:
            excluded.append(dict(source_index=idx, template=t, reason=f"first clause '{fc}' is an inspected frame")); continue
        if fc in seen:
            excluded.append(dict(source_index=idx, template=t, reason=f"first clause '{fc}' duplicates an admitted template")); continue
        seen.add(fc)
        s = abba(t)
        fill = {}
        if "[PLACE]" in s:
            fill["PLACE"] = places[ip]; ip += 1
        if "[OBJECT]" in s:
            fill["OBJECT"] = objects[io]; io += 1
        for k, v in fill.items():
            s = s.replace(f"[{k}]", v)
        admitted.append(dict(k=len(admitted) + 1, source_index=idx, source_template=t, first_clause=fc, fill=fill, template=s,
                             note=("lexical overlap with an inspected frame: the relative clause contains 'went to'"
                                   if fc.startswith("PLACE ...") else "")))
    for n in NAMES:
        assert len(tok(" " + n).input_ids) == 1, n
    for T in admitted:
        lens = {len(tok(T["template"].format(A=a, B=b)).input_ids) for a in NAMES for b in NAMES if a != b}
        assert len(lens) == 1, (T["template"], lens)
        T["n_tokens"] = lens.pop()
        prompts, j = [], 0
        while len(prompts) < 3:
            seed = 5000 + 100 * T["k"] + j; j += 1
            rng = np.random.default_rng(seed)
            rng.integers(1)
            a, b = (str(x) for x in rng.choice(NAMES, 2, replace=False))
            if any(p["A"] == a and p["B"] == b for p in prompts):
                continue
            donors = [tuple(str(x) for x in rng.choice(NAMES, 2, replace=False)) for _ in range(32)]
            prompts.append(dict(seed=seed, A=a, B=b, prompt=T["template"].format(A=a, B=b), donor_pairs=donors))
        T["prompts"] = prompts
    for T in admitted:
        for p in T["prompts"]:
            assert p["prompt"] not in {x.format(A=p["A"], B=p["B"]) for x in INSPECTED}
    pool = dict(source=SOURCE, names=NAMES, inspected_templates=INSPECTED, places_used=places, objects_used=objects,
                excluded=excluded, admitted=admitted, published=PUBLISHED, controls=CONTROLS,
                n_templates=len(admitted), n_prompts=sum(len(T["prompts"]) for T in admitted))
    txt = json.dumps(pool, indent=1)
    open(os.path.join(HERE, "prompt_pool.json"), "w").write(txt + "\n")
    print(f"admitted {pool['n_templates']} templates, {pool['n_prompts']} prompts; excluded {len(excluded)}; "
          f"sha256 {hashlib.sha256((txt + chr(10)).encode()).hexdigest()}")
    for T in admitted:
        print(f"  T{T['k']} [{T['n_tokens']} tok] {T['template']}   <- source #{T['source_index']}  {T['note']}")
        for p in T["prompts"]:
            print(f"      seed {p['seed']}: {p['prompt']}")
    for e in excluded:
        print(f"  excluded #{e['source_index']}: {e['reason']}")


if __name__ == "__main__":
    main()
