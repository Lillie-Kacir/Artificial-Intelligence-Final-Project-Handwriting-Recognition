# Development setup

## 1) Install system prerequisite (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
```

## 2) Create virtual environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

## 3) Install project dependencies

```bash
.venv/bin/pip install torch torchvision kagglehub
```

## 4) Run the training script

Full run (default):

```bash
.venv/bin/python Letter_Detection.py
```

Quick smoke test (recommended for environment validation):

```bash
EPOCHS=1 MAX_TRAIN_SAMPLES=64 MAX_VAL_SAMPLES=64 BATCH_SIZE=16 .venv/bin/python Letter_Detection.py
```

If this command prints `Epoch 1 Loss: ...`, the development environment is working.
