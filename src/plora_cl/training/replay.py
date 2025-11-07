"""Replay generation utilities."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    pipeline,
)


@dataclass
class ReplayConfig:
    """Configuration for generative replay."""

    replay_ratio: float = 0.2  # Fraction of batch to be replay samples
    min_samples_per_class: int = 10
    max_samples_per_class: int = 50
    use_pseudo_replay: bool = True  # Use template-based pseudo-replay
    generation_model: str = "gpt2"  # Model for text generation
    max_gen_length: int = 50  # Maximum length for generated text
    temperature: float = 0.7  # Generation temperature
    top_p: float = 0.9  # Nucleus sampling parameter


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
        config: Optional[ReplayConfig] = None,
    ):
        """
        Initialize the pseudo-replay generator.

        Args:
            task_configs: List of task configurations
            tokenizer: Tokenizer for encoding
            device: Device to use
            config: Replay configuration
        """
        self.task_configs = task_configs
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or ReplayConfig()

        # Load generation model if using generative replay
        if not self.config.use_pseudo_replay:
            self.generation_tokenizer = AutoTokenizer.from_pretrained(
                self.config.generation_model
            )
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token

            self.generator = pipeline(
                "text-generation",
                model=self.config.generation_model,
                tokenizer=self.generation_tokenizer,
                device=device,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )
        else:
            self.generator = None

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

    def _create_generation_prompts(self) -> Dict[str, Dict[int, str]]:
        """Create structured prompts for text generation by task and class."""
        prompts = {}

        for task_config in self.task_configs:
            task_name = task_config.name
            prompts[task_name] = {}

            if task_name == "ag_news":
                prompts[task_name] = {
                    0: "Write a news article about world events or international affairs:",
                    1: "Write a news article about sports or athletic events:",
                    2: "Write a news article about business or financial news:",
                    3: "Write a news article about technology or scientific discoveries:",
                }
            elif task_name in ["yelp_polarity", "amazon_polarity"]:
                prompts[task_name] = {
                    0: "Write a negative product review expressing disappointment:",
                    1: "Write a positive product review expressing satisfaction:",
                }
            elif task_name == "dbpedia_14":
                # Generic entity prompts
                entity_types = [
                    "company", "educational institution", "artist", "athlete",
                    "political figure", "transportation", "building", "nature",
                    "village", "animal", "plant", "album", "film", "book"
                ]
                for i in range(task_config.num_classes):
                    entity_type = entity_types[i % len(entity_types)]
                    prompts[task_name][i] = f"Write a description of a {entity_type}:"
            elif task_name == "yahoo_answers_topics":
                # Generic topic prompts
                topics = [
                    "science and mathematics", "health", "education and reference",
                    "computers and internet", "sports", "business and finance",
                    "entertainment and music", "family and relationships",
                    "politics and government", "religion and spirituality"
                ]
                for i in range(task_config.num_classes):
                    topic = topics[i % len(topics)]
                    prompts[task_name][i] = f"Write a question about {topic}:"
            else:
                # Fallback prompts
                for i in range(task_config.num_classes):
                    prompts[task_name][i] = f"Write a sample text for class {i}:"

        return prompts

    def _generate_text_with_model(self, prompt: str, num_samples: int) -> List[str]:
        """Generate text samples using the generative model."""
        if self.generator is None:
            return [f"Generated sample {i}" for i in range(num_samples)]

        generated_texts = []
        for _ in range(num_samples):
            try:
                outputs = self.generator(
                    prompt,
                    max_length=self.config.max_gen_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.generation_tokenizer.eos_token_id,
                )
                generated_text = outputs[0]["generated_text"]
                # Remove the prompt from the generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                generated_texts.append(generated_text)
            except Exception as e:
                print(f"Warning: Generation failed, using fallback: {e}")
                generated_texts.append(f"Generated sample with prompt: {prompt}")

        return generated_texts

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
            return []

        samples = []

        # Use generative model if available
        if not self.config.use_pseudo_replay and self.generator is not None:
            generation_prompts = self._create_generation_prompts()
            prompt = generation_prompts.get(task_name, {}).get(class_label, f"Write sample text for class {class_label}:")
            generated_texts = self._generate_text_with_model(prompt, num_samples)
        else:
            # Use template-based generation
            templates = self.class_templates[task_name].get(class_label, [])
            if not templates:
                return []

            generated_texts = []
            for i in range(num_samples):
                template = templates[i % len(templates)]
                if existing_texts:
                    text = template.format(existing_texts[i % len(existing_texts)])
                else:
                    # Generate more realistic placeholder text
                    placeholders = [
                        "This is an example text that demonstrates the content.",
                        "Here is a sample paragraph with relevant information.",
                        "This text contains typical content for this category.",
                        "An example of content that would fit this classification.",
                        "Sample text showing characteristics of this class.",
                    ]
                    placeholder = placeholders[i % len(placeholders)]
                    text = template.format(placeholder)
                generated_texts.append(text)

        # Tokenize all generated texts
        for text in generated_texts:
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

            # Move to device
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            tokenized["labels"] = torch.tensor([class_label], device=self.device)
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
            "config": self.config,
        }

    def load_state(self, state: Dict):
        """
        Load state from checkpoint.

        Args:
            state: State dictionary
        """
        self.class_templates = state.get("class_templates", self.class_templates)
