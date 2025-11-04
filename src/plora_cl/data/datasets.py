"""Dataset definitions for continual learning tasks."""

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    name: str
    dataset_name: str
    text_column: str
    label_column: str
    num_classes: int
    max_length: int = 512
    train_split: Optional[str] = None
    test_split: Optional[str] = None


# Task configurations matching the paper
AG_NEWS = TaskConfig(
    name="ag_news",
    dataset_name="ag_news",
    text_column="text",
    label_column="label",
    num_classes=4,
    max_length=256,
)

YELP_POLARITY = TaskConfig(
    name="yelp_polarity",
    dataset_name="yelp_polarity",
    text_column="text",
    label_column="label",
    num_classes=2,
    max_length=256,
)

AMAZON_REVIEWS = TaskConfig(
    name="amazon_polarity",
    dataset_name="amazon_polarity",
    text_column="content",
    label_column="label",
    num_classes=2,
    max_length=256,
)

DBPEDIA = TaskConfig(
    name="dbpedia_14",
    dataset_name="dbpedia_14",
    text_column="content",
    label_column="label",
    num_classes=14,
    max_length=512,
)

YAHOO_ANSWERS = TaskConfig(
    name="yahoo_answers_topics",
    dataset_name="yahoo_answers_topics",
    text_column="text",
    label_column="topic",
    num_classes=10,
    max_length=512,
)

# Sequence of tasks as per paper
TASK_SEQUENCE = [
    AG_NEWS,
    YELP_POLARITY,
    AMAZON_REVIEWS,
    DBPEDIA,
    YAHOO_ANSWERS,
]


def load_task_dataset(
    task_config: TaskConfig,
    split: str = "train",
    seed: Optional[int] = None,
) -> Dataset:
    """
    Load a dataset for a specific task.

    Args:
        task_config: Configuration for the task
        split: Dataset split to load
        seed: Random seed for reproducibility

    Returns:
        HuggingFace Dataset object
    """
    dataset = load_dataset(task_config.dataset_name, split=split)

    # Rename columns to standard names
    if task_config.text_column not in dataset.column_names:
        # Try to find text column
        text_cols = [
            col
            for col in dataset.column_names
            if col in ["text", "content", "question", "title"]
        ]
        if text_cols:
            dataset = dataset.rename_column(text_cols[0], "text")
        else:
            raise ValueError(
                f"Could not find text column in {task_config.dataset_name}"
            )

    if task_config.label_column not in dataset.column_names:
        # Try to find label column
        label_cols = [
            col
            for col in dataset.column_names
            if col in ["label", "topic", "class"]
        ]
        if label_cols:
            dataset = dataset.rename_column(label_cols[0], "label")
        else:
            raise ValueError(
                f"Could not find label column in {task_config.dataset_name}"
            )

    return dataset


def create_dataloader(
    dataset: Dataset,
    tokenizer,
    task_config: TaskConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for a task dataset.

    Args:
        dataset: HuggingFace Dataset
        tokenizer: Tokenizer for preprocessing
        task_config: Task configuration
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    from ..data.preprocessing import preprocess_function

    def tokenize(examples):
        return preprocess_function(
            examples,
            tokenizer,
            task_config.max_length,
            text_column="text",
            label_column="label",
        )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    tokenized_dataset.set_format("torch")

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


