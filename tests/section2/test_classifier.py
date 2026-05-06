from concurrent.futures import ThreadPoolExecutor

import pytest

from section2.classifier import RateLimitError, classify_sentences


class FakeClient:
    def __init__(self, batch_results=None, embedded_results=None, fail_once=False):
        self.batch_results = batch_results or []
        self.embedded_results = embedded_results or {}
        self.fail_once = fail_once
        self.batch_calls = []
        self.embedded_calls = []

    def classify_batch(self, sentences):
        self.batch_calls.append(list(sentences))
        if self.fail_once:
            self.fail_once = False
            raise RateLimitError("slow down")
        if len(sentences) == 1:
            sentence = sentences[0]
            for result in self.batch_results:
                if result["sentence"] == sentence:
                    return [result]
        return self.batch_results

    def inspect_embedded(self, sentence):
        self.embedded_calls.append(sentence)
        return self.embedded_results[sentence]


def test_normal_batch_does_not_call_embedded_agent() -> None:
    client = FakeClient(
        [
            {"sentence": "I ran.", "label": "Simple"},
            {"sentence": "I ran and she hid.", "label": "Compound"},
        ]
    )

    result = classify_sentences(["I ran.", "I ran and she hid."], client)

    assert [row["label"] for row in result] == ["Simple", "Compound"]
    assert client.embedded_calls == []
    assert result[0]["agent_path"] == ["classifier"]


def test_incomplete_routes_to_embedded_agent() -> None:
    client = FakeClient(
        [{"sentence": "because I go Home and", "label": "Incomplete"}],
        {"because I go Home and": {"found": True, "embedded_sentence": "I go Home", "final_label": "Simple"}},
    )

    result = classify_sentences(["because I go Home and"], client)

    assert result[0]["label"] == "Simple"
    assert result[0]["embedded_sentence"] == "I go Home"
    assert result[0]["original_flag"] == "Incomplete"
    assert result[0]["agent_path"] == ["classifier", "embedded"]


def test_embedded_agent_finds_none() -> None:
    client = FakeClient(
        [{"sentence": "because after the", "label": "Incomplete"}],
        {"because after the": {"found": False, "final_label": "Incomplete"}},
    )

    result = classify_sentences(["because after the"], client)

    assert result[0]["label"] == "Incomplete"
    assert result[0]["embedded_sentence"] is None


def test_embedded_agent_multiple_uses_final_label() -> None:
    client = FakeClient(
        [{"sentence": "frag", "label": "Incomplete"}],
        {"frag": {"found": True, "embedded_sentence": "I ran and she hid because it rained", "final_label": "Compound-Complex"}},
    )

    result = classify_sentences(["frag"], client)

    assert result[0]["label"] == "Compound-Complex"


def test_rate_limit_retry() -> None:
    client = FakeClient([{"sentence": "I ran.", "label": "Simple"}], fail_once=True)

    result = classify_sentences(["I ran."], client, max_retries=2, sleep_seconds=0)

    assert result[0]["label"] == "Simple"
    assert len(client.batch_calls) == 2


def test_missing_classifier_results_are_recovered_individually() -> None:
    class MissingSecondClient(FakeClient):
        def classify_batch(self, sentences):
            self.batch_calls.append(list(sentences))
            if len(sentences) > 1:
                return [{"sentence": "I ran.", "label": "Simple"}]
            return [{"sentence": sentences[0], "label": "Simple"}]

    client = MissingSecondClient()

    result = classify_sentences(["I ran.", "She hid."], client)

    assert [row["sentence"] for row in result] == ["I ran.", "She hid."]
    assert ["She hid."] in client.batch_calls


def test_rate_limit_exhaustion_raises() -> None:
    class AlwaysLimited(FakeClient):
        def classify_batch(self, sentences):
            raise RateLimitError("no")

    with pytest.raises(RateLimitError):
        classify_sentences(["I ran."], AlwaysLimited(), max_retries=1, sleep_seconds=0)


def test_concurrent_calls_do_not_share_result_state() -> None:
    client = FakeClient([{"sentence": "I ran.", "label": "Simple"}])

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: classify_sentences(["I ran."], client), range(3)))

    assert all(result[0]["sentence"] == "I ran." for result in results)
