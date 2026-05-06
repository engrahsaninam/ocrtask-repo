import time
from typing import Any, Mapping, Protocol, Sequence


class RateLimitError(Exception):
    pass


class AgentClient(Protocol):
    def classify_batch(self, sentences: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        ...

    def inspect_embedded(self, sentence: str) -> Mapping[str, Any]:
        ...


def _call_with_retry(func, max_retries: int, sleep_seconds: float):
    attempts = 0
    while True:
        try:
            return func()
        except RateLimitError:
            attempts += 1
            if attempts > max_retries:
                raise
            time.sleep(sleep_seconds * (2 ** (attempts - 1)))


def _batch(sentences: Sequence[str], size: int) -> list[list[str]]:
    return [list(sentences[i : i + size]) for i in range(0, len(sentences), size)]


def _finalize(sentence: str, classifier_result: Mapping[str, Any], client: AgentClient) -> dict:
    label = classifier_result["label"]
    row = {
        "sentence": sentence,
        "label": label,
        "embedded_sentence": None,
        "original_flag": None,
        "agent_path": ["classifier"],
    }
    if label != "Incomplete":
        return row

    embedded = client.inspect_embedded(sentence)
    row["agent_path"] = ["classifier", "embedded"]
    row["original_flag"] = "Incomplete"
    row["label"] = embedded.get("final_label", "Incomplete")
    row["embedded_sentence"] = embedded.get("embedded_sentence") if embedded.get("found") else None
    return row


def classify_sentences(
    sentences: Sequence[str],
    client: AgentClient,
    batch_size: int = 20,
    max_retries: int = 3,
    sleep_seconds: float = 0.1,
) -> list[dict]:
    final = []
    for chunk in _batch(sentences, batch_size):
        results = _call_with_retry(lambda: client.classify_batch(chunk), max_retries, sleep_seconds)
        by_sentence = {result["sentence"]: result for result in results}

        for sentence in chunk:
            if sentence not in by_sentence:
                recovered = _call_with_retry(lambda s=sentence: client.classify_batch([s]), max_retries, sleep_seconds)
                if not recovered:
                    by_sentence[sentence] = {"sentence": sentence, "label": "Incomplete"}
                else:
                    by_sentence[sentence] = recovered[0]
            final.append(_finalize(sentence, by_sentence[sentence], client))
    return final
