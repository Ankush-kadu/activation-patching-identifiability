#!/usr/bin/env bash
# Build the arXiv version of Paper A.  Every number comes from the snapshot ledger's output (claims_paperA.tex,
# produced and verified by ledger_paperA.py in the release tree); Appendix A comes from corrections_ledger.json.
#   ./build.sh            regenerate Appendix A and the figures, then compile
#   FIGS=0 ./build.sh     keep the existing figure PDFs
# Works in the repository (claims from ../repro/) and in the release tree (claims from ../).
set -euo pipefail
cd "$(dirname "$0")"
CLAIMS=../repro/claims_paperA.tex
[ -f "$CLAIMS" ] || CLAIMS=../claims_paperA.tex
cp -p "$CLAIMS" claims.tex
cmp -s "$CLAIMS" claims.tex
if [ -f ../check_corrections_ledger.py ]; then python3 ../check_corrections_ledger.py >/dev/null; fi
python3 make_appendix_a.py
if [ "${FIGS:-1}" = 1 ]; then python3 scripts/make_figures.py; fi
mkdir -p build
run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex >/dev/null; }
run; bibtex build/main >/dev/null; run; run
cp build/main.pdf main.pdf
if grep -q "undefined" build/main.log; then grep "undefined" build/main.log; exit 1; fi
echo "built main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
