from section1.eval import cer
from section4.sfs import sfs


def test_sfs_exact_match_is_one() -> None:
    assert sfs("becaus I go Home", "becaus I go Home") == 1.0


def test_sfs_penalizes_silent_correction_of_nonstandard_form() -> None:
    score = sfs("becaus I go Home", "because I go home")

    assert score < 0.7


def test_sfs_is_less_harsh_for_ordinary_ocr_error() -> None:
    correction = sfs("becaus I go Home", "because I go home")
    ordinary_error = sfs("The light was bright", "The light was brlght")

    assert ordinary_error > correction


def test_sfs_handles_empty_strings() -> None:
    assert sfs("", "") == 1.0
    assert sfs("", "text") == 0.0


def test_sfs_and_cer_give_different_signals() -> None:
    gt = "becaus I go Home"
    pred = "because I go home"

    assert cer(gt, pred) < 0.25
    assert sfs(gt, pred) < 0.7
