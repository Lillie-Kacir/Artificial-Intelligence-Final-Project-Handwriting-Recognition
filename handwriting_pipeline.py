import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from docx import Document
from spellchecker import SpellChecker


def preprocess_image_for_ocr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return binary


def extract_text(preprocessed_image: np.ndarray) -> str:
    # psm 6 expects a block of text and works well for note-style scans.
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(preprocessed_image, config=config)
    return text


def predict_words(raw_text: str) -> str:
    spell = SpellChecker()
    token_pattern = re.compile(r"[A-Za-z']+|[^A-Za-z']+")
    tokens = token_pattern.findall(raw_text)

    corrected = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z']+", token):
            candidate = token.lower()
            fixed = spell.correction(candidate)
            if fixed is None:
                fixed = candidate
            if token[0].isupper():
                fixed = fixed.capitalize()
            corrected.append(fixed)
        else:
            corrected.append(token)
    return "".join(corrected)


def save_outputs(predicted_text: str, output_stem: Path) -> tuple[Path, Path]:
    txt_path = output_stem.with_suffix(".txt")
    docx_path = output_stem.with_suffix(".docx")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(predicted_text, encoding="utf-8")

    document = Document()
    document.add_heading("Predicted Handwriting Text", level=1)
    document.add_paragraph(predicted_text)
    document.save(str(docx_path))

    return txt_path, docx_path


def run_pipeline(image_path: Path, output_stem: Path) -> tuple[str, str, Path, Path]:
    preprocessed = preprocess_image_for_ocr(image_path)
    raw_text = extract_text(preprocessed)
    predicted_text = predict_words(raw_text)
    txt_path, docx_path = save_outputs(predicted_text, output_stem)
    return raw_text, predicted_text, txt_path, docx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a handwritten image, recognize letters/words, apply "
            "word prediction/correction, and export output documents."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input handwriting image file.",
    )
    parser.add_argument(
        "--output-stem",
        default="outputs/predicted_document",
        help="Output file stem (without extension).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_stem = Path(args.output_stem)

    raw_text, predicted_text, txt_path, docx_path = run_pipeline(image_path, output_stem)

    print("Raw OCR text:")
    print(raw_text.strip())
    print("\nPredicted text:")
    print(predicted_text.strip())
    print(f"\nSaved text output: {txt_path}")
    print(f"Saved document output: {docx_path}")


if __name__ == "__main__":
    main()
