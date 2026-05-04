FROM python:3.11-slim

RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p demo_assets outputs models

CMD ["gunicorn", "app_space:app", "--bind", "0.0.0.0:8080"]
