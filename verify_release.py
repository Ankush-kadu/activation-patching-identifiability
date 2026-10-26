#!/usr/bin/env python3
"""Verify the release; --full reproduces calculations in a temporary copy."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def manifest(path, base):
    count = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        target = (base / name.lstrip("*")).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError(f"Unsafe manifest path: {name}")
        if not target.is_file():
            raise FileNotFoundError(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Hash mismatch: {name}")
        count += 1
    print(f"PASS {path.name}: {count} files", flush=True)


def run(args, cwd, env):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def full():
    for command in ("bash", "pdflatex", "bibtex", "pdftotext"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command not found: {command}")
    with tempfile.TemporaryDirectory(prefix="patching-verify-") as tmp:
        copy = Path(tmp) / "release"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", ".hypothesis",
            ".cache", "build"))
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(E0_DIR=str(copy / "e0"), PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MPLBACKEND="Agg", XDG_CACHE_HOME=str(Path(tmp) / "cache"),
                   MPLCONFIGDIR=str(Path(tmp) / "mpl"))
        run([sys.executable, "-B", "ledger_paperA.py", "--verify"], copy, env)
        if (copy / "claims_paperA.tex").read_bytes() != (copy / "paper/claims.tex").read_bytes():
            raise ValueError("Paper does not use the released claim macros")
        run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "e0/test_attribution_adequacy.py", "e0/test_heldout_semantics.py"], copy, env)
        # A new output directory: never overwrite the registered analysis.
        code = """
import pathlib, sys
sys.path.insert(0, str(pathlib.Path('e0').resolve()))
import analyse_transport_replication as analysis
out = pathlib.Path('verification_rerun').resolve()
out.mkdir()
for name in ('pilot', 'fresh'):
    (out / name).symlink_to(pathlib.Path('replication', name).resolve(), target_is_directory=True)
analysis.OUT = str(out)
analysis.analyse()
analysis.selftest()
if (out / 'analysis/analysis.json').read_bytes() != pathlib.Path('replication/analysis/analysis.json').read_bytes():
    raise ValueError('Replication analysis differs from the archived result')
print('PASS replication analysis: byte-identical')
"""
        run([sys.executable, "-B", "-c", code], copy, env)
        reference = Path(tmp) / "reference.txt"
        rebuilt = Path(tmp) / "rebuilt.txt"
        run(["pdftotext", str(ROOT / "paper/main.pdf"), str(reference)], copy, env)
        run(["bash", "build.sh"], copy / "paper", env)
        run(["pdftotext", "paper/main.pdf", str(rebuilt)], copy, env)
        if reference.read_bytes() != rebuilt.read_bytes():
            raise ValueError("Rebuilt PDF text differs from the released paper")
        print("PASS rebuilt manuscript: identical extracted text", flush=True)
    print("FULL VERIFICATION PASSED; archived inputs and outputs were not modified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    manifest(ROOT / "RELEASE_MANIFEST.sha256", ROOT)
    manifest(ROOT / "MANIFEST.sha256", ROOT)
    manifest(ROOT / "replication/CAPTURES.sha256", ROOT / "replication/fresh")
    if args.full:
        full()
    else:
        print("INTEGRITY CHECK PASSED; use --full to verify calculations and manuscript.")


if __name__ == "__main__":
    main()
