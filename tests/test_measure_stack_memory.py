from scripts.measure_stack_memory import classify_headroom


def test_fits_headroom_below_70pct():
    assert classify_headroom(10_000, budget_mb=16384) == "fits-headroom"


def test_tight_between_70_and_90pct():
    assert classify_headroom(13_000, budget_mb=16384) == "tight"


def test_doesnt_fit_at_or_above_90pct():
    assert classify_headroom(15_000, budget_mb=16384) == "doesn't-fit"
