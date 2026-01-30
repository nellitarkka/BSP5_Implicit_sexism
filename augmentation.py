
SEXISM_DEFINITION = (
    "Implicit sexism refers to subtle or indirect language that reinforces "
    "gender stereotypes, questions women's competence, or normalizes inequality."
)

CSE_CONTEXT = (
    "This text may involve gender stereotypes, biased assumptions, or unequal "
    "treatment based on gender."
)


def apply_dda(text: str) -> str:
    """
    Definition-Based Data Augmentation (DDA)
    Prepends a definition of implicit sexism to the text.
    """
    return f"{SEXISM_DEFINITION} {text}"


def apply_cse(text: str) -> str:
    """
    Contextual Semantic Expansion (CSE)
    Adds contextual framing before the text.
    """
    return f"{CSE_CONTEXT} {text}"


def augment_text(text: str) -> str:
    """
    Apply both DDA and CSE.
    """
    text = apply_dda(text)
    text = apply_cse(text)
    return text
