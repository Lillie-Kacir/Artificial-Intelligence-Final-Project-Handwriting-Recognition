# Artificial-Intelligence-Final-Project-Handwriting-Recognition

## User Access
This website is accesible via a Railway hosting. It may take some time to boot, as the site is hosted on a free plan. Also, please try to use smaller image files for faster results. (Or use the images provided in the repo). Please visit this site for easy user access:

https://artificial-intelligence-final-project-handwritin-production.up.railway.app/

Uploaded images should contain black text which should be plainly written in print on a blank white backround for the best accuracy. 

For developer information, continue on with this README.

This repository includes an end-to-end handwriting processing pipeline that:

1. Takes an input image of handwritten text.
2. Runs OCR to extract letter/word sequences.
3. Applies word prediction/correction for noisy OCR output.
4. Exports the predicted text to both `.txt` and `.docx`.

Additionally, a Flask-based web application provides a user-friendly interface for uploading images and receiving transcribed text without using the command line. The web app features a space-themed UI with drag-and-drop upload, real-time processing, and automatic file download. This can be used instead of the Railway app in case it is down.

## Testing
As this project was built through agentic engineering, there had to be structures put in place to ensure the integrity of the code could be managed. To do this, we created a small-scale demo that could be deployed for testing changes in the code that were implemented with AI. Additionally, any code that was generated with AI or troubleshooting guided by AI was carefully examined line-by-line where feasible to ensure the quality of the code. You will find the demo available below.

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

Run the web app:
.venv/bin/python app.py

Then open http://localhost:5000 in your browser.

Outputs:
- `outputs/predicted_words.txt`

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

## Artificial intelligence citation

Agentic agents such as Cursor AI and Deepseek AI were used in the collabrative sense in order to develop this product. Ideation, as well as the planning phase were conducted by human thought and then given to agentic agents in order to try to create a working demo. 
