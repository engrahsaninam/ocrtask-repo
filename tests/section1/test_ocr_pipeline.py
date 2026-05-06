from pathlib import Path

from PIL import Image, ImageDraw

from section1.ocr_pipeline import OCRRegion, extract_sentences, segment_sentences


class FakeEngine:
    def __init__(self, regions):
        self.regions = regions
        self.called = False

    def extract(self, image):
        self.called = True
        assert image.ndim == 2
        return self.regions


def _image(path: Path) -> None:
    img = Image.new("L", (120, 60), 255)
    ImageDraw.Draw(img).text((10, 20), "text", fill=0)
    img.save(path)


def test_segment_sentences_is_rule_based() -> None:
    assert segment_sentences("I ran. Then I stopped! why now?") == [
        "I ran.",
        "Then I stopped!",
        "why now?",
    ]


def test_extract_sentences_filters_metadata_and_flags_low_confidence(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    _image(path)
    engine = FakeEngine(
        [
            OCRRegion("Name: Tadeo", 0.95),
            OCRRegion("December 15 2023", 0.93),
            OCRRegion("At 7:00am Tadeo woke up.", 0.88),
            OCRRegion("He saw a shadow", 0.42),
        ]
    )

    result = extract_sentences(str(path), engine, confidence_threshold=0.6)

    assert engine.called
    assert result.sentences == ["At 7:00am Tadeo woke up.", "He saw a shadow."]
    assert result.low_confidence_regions == [OCRRegion("He saw a shadow", 0.42)]
    assert "Name:" not in result.text
    assert "December" not in result.text


def test_extract_sentences_adds_terminal_period_for_fragment(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    _image(path)
    engine = FakeEngine([OCRRegion("The house shook", 0.9)])

    result = extract_sentences(str(path), engine)

    assert result.sentences == ["The house shook."]
