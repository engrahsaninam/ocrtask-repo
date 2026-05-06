import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from section1.preprocessing import preprocess_image


@dataclass(frozen=True)
class OCRRegion:
    text: str
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    text: str
    sentences: list[str]
    low_confidence_regions: list[OCRRegion]


class OCREngine(Protocol):
    def extract(self, image: np.ndarray) -> Sequence[OCRRegion]:
        ...


class TesseractOCREngine:
    def __init__(self, psm: int = 11, lang: str = "eng", tesseract_cmd: str | None = None):
        import pytesseract

        self._pytesseract = pytesseract
        fallback = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tesseract_cmd:
            self._pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif fallback.exists():
            self._pytesseract.pytesseract.tesseract_cmd = str(fallback)
        self.psm = psm
        self.lang = lang

    def extract(self, image: np.ndarray) -> Sequence[OCRRegion]:
        pil_image = Image.fromarray(image)
        data = self._pytesseract.image_to_data(
            pil_image,
            lang=self.lang,
            output_type=self._pytesseract.Output.DICT,
            config=f"--psm {self.psm}",
        )
        regions = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            cleaned = text.strip()
            if not cleaned:
                continue
            try:
                confidence = max(0.0, float(conf) / 100.0)
            except ValueError:
                confidence = 0.0
            regions.append(OCRRegion(cleaned, confidence))
        return regions


_DATE_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[/-]\d{1,2})", re.I)
_NAME_RE = re.compile(r"^\s*(name|class|date|title)\s*:", re.I)


def is_metadata_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _NAME_RE.search(stripped):
        return True
    if _DATE_RE.search(stripped) and len(stripped.split()) <= 4:
        return True
    return False


def segment_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    parts = re.findall(r"[^.!?]+[.!?]?", cleaned)
    sentences = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        sentences.append(sentence)
    return sentences


def extract_sentences(
    image_path: str,
    engine: OCREngine,
    confidence_threshold: float = 0.6,
) -> OCRResult:
    image = preprocess_image(image_path)
    regions = list(engine.extract(image))
    kept = [region for region in regions if not is_metadata_line(region.text)]
    low_confidence = [region for region in kept if region.confidence < confidence_threshold]
    text = " ".join(region.text.strip() for region in kept if region.text.strip())
    sentences = segment_sentences(text)
    return OCRResult(text=text, sentences=sentences, low_confidence_regions=low_confidence)
