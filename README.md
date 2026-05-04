# Artificial-Intelligence-Final-Project-Handwriting-Recognition

This repository now includes an end-to-end handwriting processing pipeline that:

1. Takes an input image of handwritten text.
2. Runs OCR to extract letter/word sequences.
3. Applies word prediction/correction for noisy OCR output.
4. Exports the predicted text to both `.txt` and `.docx`.

## Quick start

Create and activate/install with your virtualenv (or use `.venv/bin/...` commands directly):

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv tesseract-ocr
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

## Train and use the CharacterCNN model

Train the CNN and save a checkpoint:

```bash
EPOCHS=1 MAX_TRAIN_SAMPLES=512 MAX_VAL_SAMPLES=256 \
  .venv/bin/python Letter_Detection.py
```

Default checkpoint path:
- `models/character_cnn.pt`

Run pipeline with CNN-assisted refinement:

```bash
.venv/bin/python handwriting_pipeline.py \
  --image path/to/handwritten_image.png \
  --output-stem outputs/predicted_words \
  --cnn-checkpoint models/character_cnn.pt
```

Disable CNN and use OCR-only mode:

```bash
.venv/bin/python handwriting_pipeline.py \
  --image path/to/handwritten_image.png \
  --output-stem outputs/predicted_words \
  --disable-cnn
```

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

## Accuracy check

Run a quick evaluation against the built-in demo reference text:

```bash
.venv/bin/python evaluate_pipeline.py
```

This prints the expected text, predicted text, and word-level accuracy.
