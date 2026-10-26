#!/usr/bin/env bash
# Clean-checkout gate for the Paper-A release tree.  Copies the release to a fresh directory, clears any path
# that could reach the parent repository, and checks, from that copy alone:
#   1. every file against MANIFEST.sha256 and every replication capture against CAPTURES.sha256;
#   2. ledger_paperA.py --verify re-derives every claim, and its claims_paperA.tex is byte-identical to the one
#      the manuscript was built from (paper/claims.tex in the repository);
#   3. the unit tests pass;
#   4. the registered replication analysis, re-run from the released captures, reproduces analysis.json exactly,
#      and its self-test passes;
#   5. no code or document mentions material outside Paper A;
#   6. the manuscript builds from this tree's own ledger output with no undefined reference, and its text equals
#      the repository build's text.
#   tools/clean_checkout_gate.sh [RELEASE] [REPO_PAPER_DIR]
set -euo pipefail
REL="${1:-/data/ai_models/explainability/release}"
REPO_PAPER="${2:-$(cd "$(dirname "$0")/../../paper" && pwd)}"
BASE=/data/ai_models/explainability/tmp
mkdir -p "$BASE"
GATE="$(mktemp -d "$BASE/gate_XXXXXX")"
cp -a "$REL"/. "$GATE"/
cd "$GATE"
unset PYTHONPATH E0_DIR
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
export XDG_CACHE_HOME="$GATE/.cache" MPLCONFIGDIR="$GATE/.cache/mpl"
echo "gate copy: $GATE"

sha256sum -c --quiet MANIFEST.sha256 && echo "[1] MANIFEST ok ($(wc -l < MANIFEST.sha256) files)"
(cd replication/fresh && sha256sum -c --quiet ../CAPTURES.sha256) && echo "[1] replication captures ok"

python3 -B ledger_paperA.py --verify | tail -1
cmp claims_paperA.tex "$REPO_PAPER/claims.tex" && echo "[2] claims_paperA.tex identical to the manuscript's claims.tex"

python3 -B -m pytest -q -p no:cacheprovider -W ignore::DeprecationWarning \
  e0/test_attribution_adequacy.py e0/test_heldout_semantics.py > pytest.log 2>&1 || { cat pytest.log; exit 1; }
echo "[3] tests: $(grep -E 'passed|failed' pytest.log | tail -1)"

mkdir -p rerun && ln -s ../replication/pilot rerun/pilot && ln -s ../replication/fresh rerun/fresh
python3 -B - <<EOF
import sys; sys.path.insert(0, "e0")
import analyse_transport_replication as A
A.OUT = "$GATE/rerun"
A.analyse()
EOF
cmp rerun/analysis/analysis.json replication/analysis/analysis.json && echo "[4] replication analysis reproduced exactly"
(cd e0 && python3 -B - <<EOF
import analyse_transport_replication as A
A.OUT = "$GATE/rerun"
A.selftest()
EOF
) | tail -1

# Venues of cited, published papers (e.g. "ICLR 2024") are allowed; any other ICLR mention is not.
if grep -rIlP "ICLR(?! 202[345])|paper2|prospective/|STUDY_REGISTRY|\bFIELD\b|\bSTRICT\b|aggregation order|anonymous supplement" \
     --include=*.py --include=*.sh --include=*.md --include=*.txt --include=*.tex \
     --exclude=clean_checkout_gate.sh --exclude=check_snapshot.py . ; then
  echo "[5] FAIL: out-of-scope reference found above"; exit 1
fi
echo "[5] no out-of-scope references in code or documents"

(cd paper && E0_DIR="$GATE/e0" ./build.sh | tail -1)
pdftotext paper/main.pdf gate.txt && pdftotext "$REPO_PAPER/main.pdf" repo.txt
cmp gate.txt repo.txt && echo "[6] manuscript text identical to the repository build"
echo "CLEAN-CHECKOUT GATE OK"
