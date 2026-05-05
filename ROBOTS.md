# ROBOTS.md

This document is for AI agents (and human contributors) working in this repository.
It describes the current project structure, responsibilities of each component, and safe modification boundaries.

## 1) Project Overview

This repository implements a handwriting recognition workflow with:

1. OCR extraction (Tesseract + OpenCV preprocessing)
2. Optional CNN-assisted character refinement (PyTorch)
3. Word-level spelling correction
4. Export to `.txt` and `.docx`
5. A Flask web interface for image upload and result download

## 2) Repository Structure (Current)

```text
/
├── app.py
├── handwriting_pipeline.py
├── character_cnn.py
├── Letter_Detection.py
├── evaluate_pipeline.py
├── demo_pipeline.py
├── README.md
├── SETUP.md
├── requirements.txt
├── Dockerfile
├── .render.yaml
├── templates/
│   └── index.html
├── demo_assets/
│   ├── demo_handwritten.png
│   ├── Ai final test 1-4.jpg
│   ├── images(1).png
│   └── handwritten-note-b93cc101f9e14c578cbba08cba66f2979ec42616.jpg
├── demo_output/
│   ├── predicted_words.txt
│   └── predicted_words.docx
├── outputs/
│   ├── predicted_words.txt
│   └── predicted_words.docx
└── models/
    └── character_cnn.pt
```

## 3) Core Modules and Responsibilities

### `handwriting_pipeline.py` (primary backend pipeline)
- Central processing entrypoint for CLI and imported use.
- Handles:
  - image preprocessing (`preprocess_image_for_ocr`)
  - OCR candidate extraction/scoring (`extract_text`)
  - optional CNN refinement (`load_cnn_bundle`, `refine_text_with_cnn`)
  - word correction (`predict_words`)
  - output writing (`save_outputs`)
- `run_pipeline(...)` is the main callable API used by other scripts.

### `app.py` (web service entrypoint)
- Flask app with:
  - `/` route returning embedded HTML page
  - `/upload` route saving input file, executing `handwriting_pipeline.py`, and returning `outputs/predicted_words.txt`
- Uses `demo_assets/` for uploads and `outputs/` for generated artifacts.

### `character_cnn.py` (model definition)
- Defines `CharacterCNN` and `CnnConfig`.
- No dataset logic; architecture only.

### `Letter_Detection.py` (training script)
- Downloads and prepares handwriting character dataset via `kagglehub`.
- Trains `CharacterCNN`.
- Saves checkpoint to `models/character_cnn.pt` (default).

### `evaluate_pipeline.py` (quality check)
- Runs `run_pipeline(...)` against demo input and expected text.
- Reports word-level accuracy for quick validation.

### `demo_pipeline.py` (demo data generation)
- Creates synthetic handwritten-like image in `demo_assets/`.
- Executes full pipeline and writes outputs in `demo_output/`.

## 4) Supporting Assets and Docs

### `templates/index.html`
- Alternate/static upload UI template.
- Note: current `app.py` serves an inline HTML string directly, not this file.

### `README.md`
- User/developer usage guide and CLI examples.

### `SETUP.md`
- Environment setup and quick validation commands.

### `requirements.txt`
- Python dependencies for runtime and training.

### `Dockerfile`
- Containerized runtime setup (installs Tesseract and Python dependencies).

### `.render.yaml`
- Render deployment config.
- Important: references `build.sh`, but no `build.sh` currently exists in repo.

## 5) Runtime Data Flow

1. Input image arrives (CLI `--image` or web upload).
2. `handwriting_pipeline.py` preprocesses image (grayscale, contrast, denoise, deskew, threshold variants).
3. Tesseract OCR runs across variants/PSM modes; best candidate selected by scoring.
4. If CNN checkpoint exists and CNN is enabled, OCR words are selectively refined.
5. Spell-based correction normalizes likely OCR mistakes.
6. Predicted text is saved to:
   - `<output_stem>.txt`
   - `<output_stem>.docx`

## 6) AI Editing Guidelines (Project-Specific)

### Safe places to modify for common tasks
- OCR and text quality improvements: `handwriting_pipeline.py`
- Model architecture changes: `character_cnn.py`
- Training pipeline changes: `Letter_Detection.py`
- Web/API behavior changes: `app.py`
- Evaluation logic: `evaluate_pipeline.py`
- User docs and setup steps: `README.md`, `SETUP.md`

### High-impact behavior to preserve
- `run_pipeline(...)` signature and return contract, unless all call sites are updated.
- Output artifact generation (`.txt` and `.docx`) expected by app/tests/workflows.
- Flask `/upload` returning downloadable text output.

### Directory usage conventions
- `demo_assets/`: sample and temporary uploaded images
- `outputs/`: default production pipeline outputs
- `demo_output/`: demo/evaluation outputs
- `models/`: model checkpoints

## 7) Known Gaps / Drift to Watch

1. `SETUP.md` references `app_space.py`, but active app entrypoint is `app.py`.
2. `.render.yaml` references `build.sh`, but `build.sh` is not present.
3. `templates/index.html` exists but is not currently used by `app.py`.

If you change any of these, update this document and the related setup docs in the same PR.
