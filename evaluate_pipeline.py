import argparse
from pathlib import Path
from typing import Iterable

from handwriting_pipeline import DEFAULT_CNN_CHECKPOINT, run_pipeline


def word_accuracy(expected: str, predicted: str) -> float:
    expected_tokens = expected.lower().split()
    predicted_tokens = predicted.lower().split()
    if not expected_tokens:
        return 0.0

    # Position-based word accuracy is simple and clear for this demo task.
    matches = 0
    for expected_token, predicted_token in zip(expected_tokens, predicted_tokens):
        if expected_token == predicted_token:
            matches += 1
    return matches / len(expected_tokens)


def evaluate(
    image: Path,
    output_stem: Path,
    expected_lines: Iterable[str],
    cnn_checkpoint: Path | None,
) -> float:
    _, predicted_text, _, _ = run_pipeline(
        image,
        output_stem,
        cnn_checkpoint=cnn_checkpoint,
    )
    expected_text = "\n".join(expected_lines).strip()
    accuracy = word_accuracy(expected_text, predicted_text.strip())
    print(f"Expected text:\n{expected_text}\n")
    print(f"Predicted text:\n{predicted_text.strip()}\n")
    print(f"Word accuracy: {accuracy:.2%}")
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline quality on the demo reference text."
    )
    parser.add_argument(
        "--disable-cnn",
        action="store_true",
        help="Disable CharacterCNN-assisted refinement for comparison.",
    )
    parser.add_argument(
        "--cnn-checkpoint",
        default=str(DEFAULT_CNN_CHECKPOINT),
        help="Path to CharacterCNN checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cnn_checkpoint = None if args.disable_cnn else Path(args.cnn_checkpoint)
    expected_demo = [
        "This is a handwritten project demo.",
        "It identifies letters and predicts words.",
        "The output should become a clean document.",
    ]
    evaluate(
        image=Path("demo_assets/demo_handwritten.png"),
        output_stem=Path("demo_output/predicted_words"),
        expected_lines=expected_demo,
        cnn_checkpoint=cnn_checkpoint,
    )
