"""Replay generation utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


@dataclass
class ReplayConfig:
    """Configuration for generative replay."""

    replay_ratio: float = 0.2  # Fraction of batch to be replay samples
    min_samples_per_class: int = 10
    max_samples_per_class: int = 50
    use_pseudo_replay: bool = True  # Use template-based pseudo-replay


class PseudoReplayGenerator:
    """
    Generate pseudo-replay samples using templates.

    This is a simplified version of generative replay that works
    with non-generative models like BERT/DistilBERT.
    """

    def __init__(
        self,
        task_configs: List,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
    ):
        """
        Initialize the pseudo-replay generator.

        Args:
            task_configs: List of task configurations
            tokenizer: Tokenizer for encoding
            device: Device to use
        """
        self.task_configs = task_configs
        self.tokenizer = tokenizer
        self.device = device
        self.class_templates = self._create_templates()

    def _create_templates(self) -> Dict[str, Dict[int, List[str]]]:
        """Create templates for each task and class."""
        templates = {}

        for task_config in self.task_configs:
            task_name = task_config.name
            templates[task_name] = {}

            # Simple templates based on task type
            if task_name == "ag_news":
                templates[task_name] = {
                    0: ["World news: {}", "International: {}", "Global: {}"],
                    1: ["Sports: {}", "Athletics: {}", "Game: {}"],
                    2: ["Business: {}", "Finance: {}", "Economy: {}"],
                    3: ["Tech: {}", "Technology: {}", "Science: {}"],
                }
            elif task_name in ["yelp_polarity", "amazon_polarity"]:
                templates[task_name] = {
                    0: ["Negative review: {}", "Bad: {}", "Poor: {}"],
                    1: ["Positive review: {}", "Good: {}", "Excellent: {}"],
                }
            elif task_name == "dbpedia_14":
                # Generic entity templates
                for i in range(task_config.num_classes):
                    templates[task_name][i] = [
                        f"Entity {i}: {{}}",
                        f"Category {i}: {{}}",
                    ]
            elif task_name == "yahoo_answers_topics":
                # Generic topic templates
                for i in range(task_config.num_classes):
                    templates[task_name][i] = [
                        f"Topic {i}: {{}}",
                        f"Question {i}: {{}}",
                    ]
            else:
                # Fallback: generic templates
                for i in range(task_config.num_classes):
                    templates[task_name][i] = [f"Class {i}: {{}}"]

        return templates

    def generate_samples(
        self,
        task_name: str,
        class_label: int,
        num_samples: int,
        existing_texts: Optional[List[str]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate pseudo-replay samples for a task and class.

        Args:
            task_name: Name of the task
            class_label: Class label
            num_samples: Number of samples to generate
            existing_texts: Existing texts to use as context

        Returns:
            List of tokenized samples
        """
        if task_name not in self.class_templates:
            # Fallback: generate empty samples
            return []

        templates = self.class_templates[task_name].get(class_label, [])
        if not templates:
            return []

        samples = []
        for i in range(num_samples):
            # Select template
            template = templates[i % len(templates)]

            # Use existing text if available
            if existing_texts:
                text = template.format(existing_texts[i % len(existing_texts)])
            else:
                # Generate placeholder text
                text = template.format(f"Sample {i}")

            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            tokenized["labels"] = torch.tensor([class_label])
            samples.append(tokenized)

        return samples

    def generate_batch_replay(
        self,
        task_name: str,
        num_classes: int,
        batch_size: int,
        replay_ratio: float = 0.2,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate a batch of replay samples.

        Args:
            task_name: Name of the task
            num_classes: Number of classes
            batch_size: Size of the batch
            replay_ratio: Ratio of replay samples

        Returns:
            List of replay samples
        """
        num_replay = int(batch_size * replay_ratio)
        samples_per_class = max(1, num_replay // num_classes)

        replay_samples = []
        for class_label in range(num_classes):
            class_samples = self.generate_samples(
                task_name, class_label, samples_per_class
            )
            replay_samples.extend(class_samples)

        # Trim to exact number needed
        return replay_samples[:num_replay]
    
    def get_state(self) -> Dict:
        """
        Get state for checkpointing.
        
        Returns:
            State dictionary
        """
        return {
            "class_templates": self.class_templates,
        }
    
    def load_state(self, state: Dict):
        """
        Load state from checkpoint.
        
        Args:
            state: State dictionary
        """
        self.class_templates = state.get("class_templates", self.class_templates)


