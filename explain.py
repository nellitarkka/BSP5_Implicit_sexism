from typing import List, Tuple


def generate_explanation(
    text: str,
    prediction: int,
    retrieved_items: List[Tuple[str, float]]
) -> str:
    """
    Generate a natural language explanation for the prediction.

    prediction:
        1 -> Sexist
        0 -> Not Sexist
    """

    label_str = "Sexist" if prediction == 1 else "Not Sexist"

    explanation_lines = []
    explanation_lines.append(f"Input text: \"{text}\"")
    explanation_lines.append(f"Prediction: {label_str}")
    explanation_lines.append("")

    if prediction == 1:
        explanation_lines.append(
            "Explanation: The sentence was classified as sexist because it aligns "
            "with known patterns of implicit sexism, as shown by the following "
            "retrieved definitions and examples:"
        )
    else:
        explanation_lines.append(
            "Explanation: The sentence was classified as not sexist because it does "
            "not strongly align with known patterns of implicit sexism."
        )

    explanation_lines.append("")

    for idx, (item, score) in enumerate(retrieved_items, start=1):
        explanation_lines.append(
            f"{idx}. Retrieved reference (similarity={score:.2f}): {item}"
        )

    return "\n".join(explanation_lines)
