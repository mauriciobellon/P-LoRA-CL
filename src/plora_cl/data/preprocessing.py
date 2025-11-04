"""Text preprocessing utilities."""

from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, List]:
    """
    Preprocess text examples for tokenization.

    Args:
        examples: Dictionary of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Tokenized examples
    """
    texts = examples[text_column]

    # Tokenize texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    # Add labels if present
    if label_column in examples:
        tokenized["labels"] = examples[label_column]

    return tokenized


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    import re

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip whitespace
    text = text.strip()
    return text


