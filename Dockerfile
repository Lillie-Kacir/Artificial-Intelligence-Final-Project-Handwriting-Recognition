FROM python:3.11-slim

# Install Tesseract and OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p demo_assets outputs models

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]