#!/usr/bin/env bash
# Assemble the Paper-A-only release tree from the snapshot (code) and the project's e0/ (data), then verify
# every file against MANIFEST.sha256.  The release tree is self-contained: its scripts import only each other.
#
#   tools/make_release.sh [DEST]      (default DEST=/data/ai_models/explainability/release)
set -euo pipefail
REPRO="$(cd "$(dirname "$0")/.." && pwd)"
SRC_E0="${SRC_E0:-$REPRO/../../e0}"
DEST="${1:-/data/ai_models/explainability/release}"

mkdir -p "$DEST/e0" "$DEST/tools"
cp -p "$REPRO"/claims_paperA.json "$REPRO"/claims_paperA.tex "$DEST/" 2>/dev/null || true   # ledger output the paper builds from
cp -p "$REPRO"/ledger_paperA.py "$REPRO"/ledger_additions.py "$REPRO"/requirements.txt "$REPRO"/DATA_FILES.txt "$REPRO"/MANIFEST.sha256 "$DEST/"
[ -f "$REPRO/README.md" ] && cp -p "$REPRO/README.md" "$DEST/"
cp -p "$REPRO"/tools/*.py "$REPRO"/tools/*.sh "$DEST/tools/"
cp -p "$REPRO"/e0/*.py "$DEST/e0/"
[ -d "$REPRO/artifacts" ] && cp -p "$REPRO"/artifacts/* "$DEST/e0/"     # artifacts new in the arXiv version
mkdir -p "$DEST/prereg" && cp -p "$REPRO"/../prereg/* "$DEST/prereg/"   # frozen preregistration and prompt pool
while IFS= read -r f; do
  [ -n "$f" ] && cp -p "$SRC_E0/$f" "$DEST/e0/$f"
done < "$REPRO/DATA_FILES.txt"
(cd "$DEST" && sha256sum -c --quiet MANIFEST.sha256)
echo "release assembled at $DEST and verified against MANIFEST.sha256 ($(wc -l < "$REPRO/MANIFEST.sha256") files)"

# Preregistered replication: the registered outputs and the fresh captures they were computed from.
REP_SRC="${REP_SRC:-/data/ai_models/explainability/replication_transport}"
if [ -d "$REP_SRC/fresh" ]; then
  mkdir -p "$DEST/replication/pilot" "$DEST/replication/fresh" "$DEST/replication/analysis"
  cp -p "$REP_SRC/pilot/pilot_report.json" "$DEST/replication/pilot/"
  cp -p "$REP_SRC"/fresh/*.npz "$REP_SRC/fresh/manifest.json" "$DEST/replication/fresh/"
  cp -p "$REPRO/../replication/analysis.json" "$DEST/replication/analysis/analysis.json"
  cp -p "$REPRO/../replication/CAPTURES.sha256" "$REPRO/../replication/fresh_run.log" "$DEST/replication/"
  (cd "$DEST/replication/fresh" && sha256sum -c --quiet ../CAPTURES.sha256)
  cmp -s "$REPRO/../replication/manifest.json" "$DEST/replication/fresh/manifest.json"
  cmp -s "$REPRO/../replication/pilot_report.json" "$DEST/replication/pilot/pilot_report.json"
  echo "replication: $(ls "$DEST/replication/fresh" | grep -c '\.npz$') captures verified against CAPTURES.sha256"
fi

# Correction ledger and the manuscript sources (the paper builds from this tree's own ledger output).
PAPER="$REPRO/../paper"
if [ -d "$PAPER" ]; then
  cp -p "$REPRO/../corrections_ledger.json" "$REPRO/../CORRECTIONS_LEDGER.md" "$DEST/"
  mkdir -p "$DEST/paper/figures" "$DEST/paper/scripts"
  cp -p "$PAPER"/{main.tex,appendix.tex,appendix_a_display.py,make_appendix_a.py,refs.bib,neurips_2026.sty,build.sh,main.pdf} "$DEST/paper/"
  cp -p "$PAPER"/figures/*.pdf "$DEST/paper/figures/"
  cp -p "$PAPER"/scripts/make_figures.py "$DEST/paper/scripts/"
  echo "paper sources copied to $DEST/paper"
fi
