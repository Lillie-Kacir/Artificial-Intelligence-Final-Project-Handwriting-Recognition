# Artificial-Intelligence-Final-Project-Handwriting-Recognition

This repository now includes an end-to-end handwriting processing pipeline that:

1. Takes an input image of handwritten text.
2. Runs OCR to extract letter/word sequences.
3. Applies word prediction/correction for noisy OCR output.
4. Exports the predicted text to both `.txt` and `.docx`.

## Quick start

Create and activate/install with your virtualenv (or use `.venv/bin/...` commands directly):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Run the pipeline:

```bash
.venv/bin/python handwriting_pipeline.py \
  --image path/to/handwritten_image.png \
  --output-stem outputs/predicted_words
```

Outputs:
- `outputs/predicted_words.txt`
- `outputs/predicted_words.docx`

## Demo

You can generate a sample handwritten-style image and run the full pipeline:

```bash
.venv/bin/python demo_pipeline.py
.venv/bin/python handwriting_pipeline.py \
  --image sample_handwriting.png \
  --output-stem demo_output/predicted_words
```

This creates:
- `sample_handwriting.png`
- `demo_output/predicted_words.txt`
- `demo_output/predicted_words.docx`
