"""Write MANIFEST.sha256 for the Paper-A snapshot: every snapshot script plus every data file in DATA_FILES.txt.

Script hashes are taken from the snapshot itself; data hashes from the source data directory (default: the
project's e0/), so the release assembler can verify the copies.  Format is `sha256sum -c` compatible.

    python tools/make_manifest.py [--src-e0 PATH]
"""
import argparse, hashlib, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--src-e0", default=os.path.join(HERE, "..", "..", "e0"))
A = ap.parse_args()
lines = []
code = ["ledger_paperA.py", "ledger_additions.py", "requirements.txt", "DATA_FILES.txt"] + \
       sorted("e0/" + f for f in os.listdir(os.path.join(HERE, "e0")) if f.endswith(".py")) + \
       sorted("tools/" + f for f in os.listdir(os.path.join(HERE, "tools")) if f.endswith((".py", ".sh")))
for rel in code:
    lines.append(f"{sha(os.path.join(HERE, rel))}  {rel}")
data = [l.strip() for l in open(os.path.join(HERE, "DATA_FILES.txt")) if l.strip()]
for f in data:
    lines.append(f"{sha(os.path.join(A.src_e0, f))}  e0/{f}")
art = sorted(os.listdir(os.path.join(HERE, "artifacts"))) if os.path.isdir(os.path.join(HERE, "artifacts")) else []
for f in art:
    lines.append(f"{sha(os.path.join(HERE, 'artifacts', f))}  e0/{f}")
open(os.path.join(HERE, "MANIFEST.sha256"), "w").write("\n".join(lines) + "\n")
print(f"MANIFEST.sha256: {len(code)} code files, {len(data)} data files")
