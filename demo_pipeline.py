from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from handwriting_pipeline import run_pipeline


def create_demo_handwritten_like_image(path: str) -> None:
    width, height = 1200, 500
    image = Image.new("RGB", (width, height), color=(250, 250, 245))
    draw = ImageDraw.Draw(image)

    text = (
        "Ths is a handwriten project demo.\n"
        "It identifyes letters and predicts words.\n"
        "The output shuld become a clean document."
    )

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 52)
    except OSError:
        font = ImageFont.load_default()

    draw.multiline_text((60, 90), text, fill=(40, 40, 40), font=font, spacing=16)
    image.save(path)


if __name__ == "__main__":
    demo_image = Path("demo_assets/demo_handwritten.png")
    output_stem = Path("demo_output/predicted_words")
    demo_image.parent.mkdir(parents=True, exist_ok=True)

    create_demo_handwritten_like_image(str(demo_image))
    print(f"Created {demo_image}")

    raw_text, predicted_text, txt_path, docx_path = run_pipeline(demo_image, output_stem)
    print("\nRaw OCR text:")
    print(raw_text.strip())
    print("\nPredicted text:")
    print(predicted_text.strip())
    print(f"\nSaved text output: {txt_path}")
    print(f"Saved document output: {docx_path}")
