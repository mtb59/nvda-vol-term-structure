import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from black_scholes.data import synthetic_bloomberg_snapshot
from black_scholes.validate_bloomberg import run_validation, summarise_validation


def test_validation_runs_and_returns_expected_columns():
    snap = synthetic_bloomberg_snapshot()
    results = run_validation(snap)
    assert not results.empty
    for col in ["iv_diff", "delta_diff", "gamma_diff", "vega_diff", "theta_diff"]:
        assert col in results.columns


def test_validation_detects_american_exercise_signature_when_present():
    snap_biased = synthetic_bloomberg_snapshot(american_exercise_bias=True, seed=99)
    summary_biased = summarise_validation(run_validation(snap_biased))
    assert summary_biased["put_vs_call_divergence_ratio"] > 1.3, (
        "Expected put delta divergence to exceed call delta divergence when "
        "an early-exercise premium is present in the synthetic Bloomberg data"
    )


def test_validation_shows_no_signature_when_absent():
    snap_clean = synthetic_bloomberg_snapshot(american_exercise_bias=False, seed=99)
    summary_clean = summarise_validation(run_validation(snap_clean))
    assert summary_clean["put_vs_call_divergence_ratio"] < 1.3, (
        "Expected roughly symmetric put/call divergence when no structural "
        "bias was injected into the synthetic Bloomberg data"
    )


def run_all():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
