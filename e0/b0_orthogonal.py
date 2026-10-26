"""Orthogonal-reparameterisation check behind the representation table's omission of degree 3 (Appendix G).
DESCRIPTIVE; one previously inspected capture; a configuration-specific diagnostic, not a portable constant.

mech_classes.py's own B0 check multiplies an orthonormal matrix by column scales (a non-orthogonal map) and then
refits the chart, which can change the induced function class. Here the raw representation of the first actsg
capture is rotated by orthogonal Q_s = qr(default_rng(s).normal((p, p)))[0], s = 0, 1, 2. The PCA chart is
rotation-equivariant, so the induced class is unchanged and any change in v_S is numerical. Every value is computed
in one process with the frozen run_file (k = 4, degrees 1-3); each rotated value is compared with the unrotated base
from the same process. The base is also recomputed with the BLAS thread pool limited to one thread, and compared
with the stored mech_classes.json value, to record how the degree-3 value depends on the BLAS configuration.

    python3 b0_orthogonal.py [--e0 DIR] [--out PATH]      # -> b0_orthogonal.json
"""
import argparse, glob, json, os, platform, sys
import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mech_classes import run_file      # noqa: E402

STATUS = "DESCRIPTIVE: orthogonal-reparameterisation diagnostic on one previously inspected capture"
K, DEGS, SEEDS = 4, (1, 2, 3), (0, 1, 2)


def vS(res):
    return {str(d): float(res[str(d)]["vS_on_Sblind"]) for d in DEGS}


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def blas_info():
    try:
        from threadpoolctl import threadpool_info
        keep = ("internal_api", "version", "num_threads", "threading_layer")
        return [{k: v for k, v in d.items() if k in keep} for d in threadpool_info()]
    except ImportError:
        return "threadpoolctl not installed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0", default=HERE, help="directory holding the actsg captures and mech_classes.json")
    ap.add_argument("--out", default=os.path.join(HERE, "b0_orthogonal.json"))
    A = ap.parse_args()
    f = sorted(glob.glob(os.path.join(A.e0, "actsg_*_*_*.npz")))[0]
    p = int(np.load(f)["A_single"].shape[1])

    base = vS(run_file(f, {"raw": lambda X: X}, K, DEGS)["raw"])
    rows = {}
    for s in SEEDS:
        Q = np.linalg.qr(np.random.default_rng(s).normal(size=(p, p)))[0]
        assert np.abs(Q.T @ Q - np.eye(p)).max() < 1e-10
        rot = vS(run_file(f, {"raw": lambda X, Q=Q: X @ Q.T}, K, DEGS)["raw"])
        rows[str(s)] = {d: dict(vS=rot[d], abs_diff=abs(rot[d] - base[d]), rel_diff=rel(rot[d], base[d])) for d in rot}
    r = lambda d: [rows[str(s)][str(d)]["rel_diff"] for s in SEEDS]

    single = None
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            single = vS(run_file(f, {"raw": lambda X: X}, K, DEGS)["raw"])
    except ImportError:
        pass
    stored = None
    mp = os.path.join(A.e0, "mech_classes.json")
    if os.path.exists(mp):
        stored = {d: float(v["vS_on_Sblind"]) for d, v in json.load(open(mp))["raw"]["B0"]["base"].items()}

    out = dict(status=STATUS, file=os.path.basename(f), k=K, degrees=list(DEGS), seeds=list(SEEDS), p=p,
               construction="Q_s = qr(default_rng(s).normal((p, p)))[0]; representation X -> X Q_s^T; chart refitted",
               base=base, orthogonal=rows,
               d1_rel_max=max(r(1)), d2_rel_max=max(r(2)), d3_rel_min=min(r(3)), d3_rel_max=max(r(3)),
               base_single_thread=single,
               base_single_thread_rel_diff=None if single is None else {d: rel(single[d], base[d]) for d in base},
               stored_mech_classes_base=stored,
               base_rel_diff_vs_stored=None if stored is None else {d: rel(base[d], stored[d]) for d in base},
               environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                blas=blas_info(),
                                thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                           "MKL_NUM_THREADS")}),
               note="degrees 1-2 are invariant under orthogonal reparameterisation to numerical precision; degree 3 is "
                    "not, and it also changes with the BLAS thread configuration (base_single_thread); "
                    "configuration-specific diagnostic, not a portable constant")
    json.dump(out, open(A.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("d1_rel_max", "d2_rel_max", "d3_rel_min", "d3_rel_max",
                                          "base_single_thread_rel_diff", "base_rel_diff_vs_stored")}, indent=1))
    print(f"saved -> {A.out}")


if __name__ == "__main__":
    main()
