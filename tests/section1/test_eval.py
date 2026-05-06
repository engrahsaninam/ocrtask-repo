from section1.eval import cer, composite_score, evaluate_samples, sentence_f1, wer


def test_cer_uses_character_edit_distance() -> None:
    assert cer("kitten", "sitten") == 1 / 6


def test_wer_uses_word_edit_distance() -> None:
    assert wer("I like coffee", "I love coffee") == 1 / 3


def test_sentence_f1_matches_token_overlap_threshold() -> None:
    gt = "I woke early. The bright light shook the house."
    pred = "I woke up early. The house shook."

    assert sentence_f1(gt, pred) == 1.0


def test_composite_score_rewards_low_error_and_sentence_recovery() -> None:
    assert composite_score(0.1, 0.2, 0.5) == 0.75


def test_evaluate_samples_returns_rows_and_aggregate() -> None:
    rows, aggregate = evaluate_samples(
        [
            {"sample_id": "a", "ground_truth": "hello world.", "predicted": "hello world."},
            {"sample_id": "b", "ground_truth": "cat sat.", "predicted": "cat."},
        ]
    )

    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert aggregate["samples"] == 2
    assert 0 <= aggregate["composite"] <= 1
