#!/bin/bash
set -e

# Update and install tesseract
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng

# Verify installation
tesseract --version

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
