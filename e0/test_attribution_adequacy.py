"""Regression tests for attribution_adequacy.py / attribution_adequacy.json.

    python -m pytest -q test_attribution_adequacy.py
"""
import glob, json, os
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "attribution_adequacy.json")


@pytest.fixture(scope="module")
def A():
    return json.load(open(OUT))


def test_status_is_exploratory(A):
    assert A["status"].startswith("EXPLORATORY")
    assert A["numerical_floor"] is None


def test_reproduces_misattribution_fourpoint(A):
    ref = {os.path.basename(r["file"]): r for r in json.load(open(os.path.join(HERE, "misattribution_fourpoint.json")))["rows"]}
    rows = A["groups"]["published"]["rows"]
    assert len(rows) == 31
    for r in rows:
        for fam in ("S", "J"):
            got = r[f"d3_{fam}"]["env_med"] / r["swing"]
            assert abs(got - ref[r["file"]][f"med_{fam}"]) <= 1e-12 * max(1.0, abs(ref[r["file"]][f"med_{fam}"]))


def test_model_contrast_equals_D_c():
    """The fitted cubic's four-point contrast at the EXECUTED corners equals D c (additive terms cancel)."""
    import misattribution as M
    import misattribution_fourpoint as F
    import fourpoint_modelfree as FM
    import response_diagnostics as RD
    f = sorted(glob.glob(os.path.join(HERE, "qchart_*.npz")))[0]
    d = np.load(f)
    T, corners, a0b0, *_ = FM.quartets(d)
    Zs, Ys, Zj = d["Zs"], d["Ys"], d["Zjm"]
    allZ = np.vstack([Zs, Zj]); mu, sd = allZ.mean(0), allZ.std(0) + 1e-12
    W = lambda Z: (Z - mu) / sd
    with RD.working_class(3) as (add, cross):
        pred = RD.working_model(Zs, Ys, W, add, cross)
        mc = FM.model_contrast(pred, corners, a0b0)
        c, _ = M.fit(Zs, Ys, W)
        Dc = F.four_point(W, Zj, *a0b0) @ c
    assert np.allclose(mc, Dc, rtol=1e-6, atol=1e-8 * np.max(np.abs(Dc)))


def test_executed_corners_are_the_contrast_corners(A):
    for g in ("published", "controls"):
        for r in A["groups"][g]["rows"]:
            a = r["audit"]
            assert a["joint_a_vs_clamped"] < 1e-5 and a["joint_b_vs_nm_arm"] < 1e-5
            assert a["clamped_b_spread"] < 1e-9 and a["nm_arm_a_spread"] < 1e-9


def test_coverage_logic(A):
    s = A["groups"]["published"]["summary"]
    rows = A["groups"]["published"]["rows"]
    for deg in A["degrees"]:
        for fam in ("S", "J"):
            k = f"d{deg}_{fam}"
            assert s[k]["cover_upper_all"] == sum(r[k]["cover_upper_all"] for r in rows)
            # |T - c| <= r implies |T| <= |c| + r, so the interval share can never exceed the upper-bound share
            assert all(r[k]["cover_interval_share"] <= r[k]["cover_upper_share"] + 1e-15 for r in rows)


def test_rebuilds_and_pairing(A):
    assert A["qc_summary"]["all_match"]
    assert A["paired_published_vs_control"]["all_cells_share_swing"]
    assert A["induction_detection"]["n"] == 18
