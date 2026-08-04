import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from datasets import load_dataset
from pydantic import Field
from transformers import AutoTokenizer

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class QuantumHybridConfig(BaseEnvConfig):
    """Configuration for the Quantum-Classical Hybrid Environment."""

    # Quantum circuit parameters
    n_qubits: int = Field(8, description="Number of qubits in the quantum circuit")
    n_layers: int = Field(
        3, description="Number of quantum layers to use in the circuit"
    )

    # Dataset parameters
    dataset_name: str = Field(
        "wikitext", description="Dataset to use for training/evaluation"
    )
    dataset_config: str = Field(
        "wikitext-2-raw-v1", description="Dataset configuration"
    )
    sequence_length: int = Field(256, description="Length of sequences to process")

    # Base model parameters
    base_model_name: str = Field(
        "gpt2", description="Base model for hybrid experiments"
    )

    # Training parameters
    perplexity_weight: float = Field(
        0.7, description="Weight for perplexity in scoring"
    )
    quantum_weight: float = Field(
        0.3, description="Weight for quantum-specific metrics"
    )


class QuantumTextAnalyzer:
    """Standalone quantum analyzer for text coherence measurement."""

    def __init__(self, n_qubits=6):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def text_analysis_circuit(text_features):
            # Embed text features as quantum states
            for i in range(self.n_qubits):
                qml.RY(text_features[i], wires=i)

            # Create entanglement patterns
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Ring closure
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Additional entanglement for complex analysis
            for i in range(0, self.n_qubits - 2, 2):
                qml.CNOT(wires=[i, i + 2])

            # Measure coherence
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = text_analysis_circuit

    def analyze_text(self, text: str) -> float:
        """Analyze text and return quantum coherence score."""
        try:
            # Extract text features
            text_len = min(len(text), 200) / 200.0
            word_count = len(text.split()) / 100.0 if text.split() else 0.0
            char_diversity = len(set(text.lower())) / 26.0 if text else 0.0
            avg_word_len = (
                np.mean([len(word) for word in text.split()]) / 15.0
                if text.split()
                else 0.0
            )

            # Additional features
            punctuation_ratio = sum(1 for c in text if c in ".,!?;:") / max(
                len(text), 1
            )
            uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

            # Encode as quantum features
            features = [
                text_len * np.pi,
                word_count * np.pi,
                char_diversity * np.pi,
                avg_word_len * np.pi,
                punctuation_ratio * np.pi,
                uppercase_ratio * np.pi,
            ]

            # Run quantum analysis
            measurements = self.circuit(features)

            # Calculate coherence as normalized measurement variance
            coherence = np.var(measurements) / (np.var(measurements) + 0.1)
            return float(np.clip(coherence, 0.0, 1.0))

        except Exception as e:
            print(f"Quantum text analysis error: {e}")
            # Fallback to simple heuristic
            return min(1.0, len(text) / 100.0) * 0.7 + 0.2


class QuantumHybridEnv(BaseEnv):
    """Environment for training and evaluating quantum-classical hybrid models."""

    def __init__(
        self,
        config: QuantumHybridConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config
        self.metrics_buffer = {
            "perplexity": [],
            "quantum_coherence": [],
            "combined_score": [],
            "quantum_variance": [],
        }

        # Initialize eval_metrics list (required for BaseEnv)
        self.eval_metrics = []

        # Initialize quantum text analyzer
        self.quantum_analyzer = QuantumTextAnalyzer(n_qubits=config.n_qubits)

        # Track training iteration
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[QuantumHybridConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        config = QuantumHybridConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=4,
            use_wandb=True,
            max_num_workers=-1,
            rollout_server_url="http://localhost:8000",  # Atropos server
            total_steps=20,
            batch_size=-1,
            max_token_length=2048,
            data_path_to_save_groups="data/quantum_hybrid.jsonl",
            n_qubits=8,
            n_layers=3,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            sequence_length=256,
            base_model_name="gpt2",
            perplexity_weight=0.7,
            quantum_weight=0.3,
        )

        # The server config here tells Atropos to route to your vLLM server
        # This should match whatever model name the Atropos server expects
        server_configs = [
            APIServerConfig(
                model_name="hermes-3-8b",  # This model name should be registered in Atropos
                base_url="http://localhost:9001/v1",  # Your vLLM server
                api_key="x",  # Placeholder for local server
                timeout=300,
                num_max_requests_at_once=8,
                num_requests_for_eval=4,
                health_check=False,
                server_type="openai",
                n_kwarg_is_ignored=False,
                rolling_buffer_length=100,
            ),
        ]

        return config, server_configs

    async def setup(self):
        """Set up the environment, including loading datasets."""
        print(
            f"Setting up QuantumHybridEnv with quantum parameters: "
            f"{self.config.n_qubits} qubits, {self.config.n_layers} layers"
        )

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split="train",
                streaming=True,
            )
            self.dataset = dataset.take(10000)  # Take first 10k examples

            eval_dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split="validation",
                streaming=True,
            )
            self.eval_dataset = eval_dataset.take(100)  # Take first 100 for eval

            print(
                f"Loaded dataset: {self.config.dataset_name}/{self.config.dataset_config}"
            )

        except Exception as e:
            print(f"Failed to load dataset: {e}")
            # Create a mock dataset for testing
            self.dataset = [
                {
                    "text": f"Sample text {i} for quantum analysis. This text demonstrates complex linguistic patterns."
                }
                for i in range(1000)
            ]
            self.eval_dataset = [
                {"text": f"Eval text {i} with various complexity levels."}
                for i in range(100)
            ]
            print("Using mock dataset")

        # Convert to lists for easier iteration
        if hasattr(self.dataset, "__iter__"):
            self.train_examples = list(self.dataset)
        else:
            self.train_examples = self.dataset

        if hasattr(self.eval_dataset, "__iter__"):
            self.eval_examples = list(self.eval_dataset)
        else:
            self.eval_examples = self.eval_dataset

        print(f"Loaded {len(self.train_examples)} training examples")
        print(f"Loaded {len(self.eval_examples)} evaluation examples")

    async def get_next_item(self):
        """Get the next training item from the dataset."""
        # Get next instance from the dataset
        if self.iter >= len(self.train_examples):
            self.iter = 0  # Reset to beginning

        data_point = self.train_examples[self.iter]
        self.iter += 1

        # Process text data
        if isinstance(data_point, dict) and "text" in data_point:
            text = data_point["text"]
        else:
            text = str(data_point)

        # Truncate text to reasonable length
        text = text[: self.config.sequence_length * 4]  # Allow for some context

        # Create a simple continuation task
        words = text.split()
        if len(words) > 10:
            # Split at a random point for continuation
            split_idx = random.randint(5, min(len(words) - 5, 20))
            prompt_text = " ".join(words[:split_idx])
            target_text = " ".join(words[split_idx : split_idx + 20])
        else:
            prompt_text = text[: len(text) // 2]
            target_text = text[len(text) // 2 :]

        # Convert to messages format for Atropos
        user_msg = {"role": "user", "content": f"Continue this text: {prompt_text}"}

        # Return as (messages, target) tuple
        return ([user_msg], target_text)

    async def collect_trajectories(self, item: Tuple) -> Tuple[ScoredDataGroup, List]:
        """Generate and collect model responses for scoring."""
        messages, target_text = item

        print(f"Generating completions for: {messages[0]['content'][:50]}...")

        to_score = []

        # Generate multiple completions with different temperatures
        temperatures = [0.6, 0.8, 1.0, 1.2][: self.config.group_size]

        for i, temp in enumerate(temperatures):
            try:
                # Use the Atropos server to generate completions
                completion = await self.server.completion(
                    prompt=self.tokenizer.apply_chat_template(messages, tokenize=False),
                    n=1,  # Generate one completion at a time
                    max_tokens=min(100, self.config.max_token_length // 4),
                    temperature=temp,
                )

                completion_text = completion.choices[0].text.strip()
                print(
                    f"Generated completion {i+1} (T={temp}): {completion_text[:50]}..."
                )

                # Create trajectory messages
                trajectory_messages = messages.copy()
                trajectory_messages.append(
                    {"role": "assistant", "content": completion_text}
                )

                # Add to scoring queue
                to_score.append(
                    (
                        tuple([frozenset(msg.items()) for msg in trajectory_messages]),
                        target_text,
                        completion_text,
                    )
                )

            except Exception as e:
                print(f"Error generating completion {i+1} with temperature {temp}: {e}")
                # Create a mock completion as fallback
                mock_text = (
                    f"Mock quantum-enhanced response {i+1}: This demonstrates "
                    f"coherent language generation with temperature {temp}."
                )
                trajectory_messages = messages.copy()
                trajectory_messages.append({"role": "assistant", "content": mock_text})
                to_score.append(
                    (
                        tuple([frozenset(msg.items()) for msg in trajectory_messages]),
                        target_text,
                        mock_text,
                    )
                )

        # Score the generated trajectories
        scored_data = await self.score(to_score)

        return scored_data, []

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        """Score model outputs based on perplexity and quantum metrics."""
        if not rollout_group_data:
            return None

        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []

        print(f"Scoring {len(rollout_group_data)} completions...")

        for item in rollout_group_data:
            frozen_messages, target_text, completion_text = item

            # Convert frozen messages back to regular format
            messages = [dict(frozen_msg) for frozen_msg in frozen_messages]

            # Convert messages to tokens
            try:
                tokenized = tokenize_for_trainer(self.tokenizer, messages)
                tokens = tokenized["tokens"]
                masks = tokenized["masks"]
            except Exception as e:
                print(f"Tokenization error: {e}")
                continue

            # Calculate text similarity score (proxy for perplexity)
            completion_words = set(completion_text.lower().split())
            target_words = set(target_text.lower().split())

            # Jaccard similarity
            if completion_words and target_words:
                intersection = len(completion_words & target_words)
                union = len(completion_words | target_words)
                similarity_score = intersection / union if union > 0 else 0.0
            else:
                similarity_score = 0.0

            # Length penalty/bonus
            len_ratio = len(completion_text) / max(len(target_text), 1)
            len_score = 1.0 - abs(1.0 - len_ratio) if len_ratio > 0 else 0.0

            # Combined perplexity-like score
            perplexity_score = 0.7 * similarity_score + 0.3 * len_score

            # Quantum coherence analysis
            quantum_coherence = self.quantum_analyzer.analyze_text(completion_text)

            # Calculate quantum variance for additional insight
            quantum_variance = (
                abs(quantum_coherence - 0.5) * 2
            )  # Distance from maximum entropy

            # Combined score using weighted sum
            combined_score = (
                self.config.perplexity_weight * perplexity_score
                + self.config.quantum_weight * quantum_coherence
            )

            print(
                f"  Similarity: {similarity_score:.3f}, Quantum: {quantum_coherence:.3f}, "
                f"Combined: {combined_score:.3f}"
            )

            # Update metrics buffer
            self.metrics_buffer["perplexity"].append(perplexity_score)
            self.metrics_buffer["quantum_coherence"].append(quantum_coherence)
            self.metrics_buffer["combined_score"].append(combined_score)
            self.metrics_buffer["quantum_variance"].append(quantum_variance)

            # Store data for training (scale to [-1, 1] range)
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(2 * combined_score - 1)

        return scores if scores["scores"] else None

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on a test set."""
        print("Running quantum-enhanced evaluation...")
        eval_scores = []
        quantum_scores = []

        # Get evaluation examples
        eval_examples = self.eval_examples[: min(5, len(self.eval_examples))]

        for example in eval_examples:
            try:
                # Process text
                if isinstance(example, dict) and "text" in example:
                    text = example["text"]
                else:
                    text = str(example)

                # Create continuation task
                words = text.split()
                if len(words) > 8:
                    split_idx = len(words) // 2
                    prompt_text = " ".join(words[:split_idx])
                    target_text = " ".join(words[split_idx:])
                else:
                    prompt_text = text[: len(text) // 2]
                    target_text = text[len(text) // 2 :]

                # Create messages
                messages = [
                    {"role": "user", "content": f"Continue this text: {prompt_text}"}
                ]

                # Generate completion
                completion = await self.server.completion(
                    prompt=self.tokenizer.apply_chat_template(messages, tokenize=False),
                    n=1,
                    max_tokens=50,
                    temperature=0.7,
                    split="eval",
                )

                completion_text = completion.choices[0].text.strip()

                # Calculate metrics
                completion_words = set(completion_text.lower().split())
                target_words = set(target_text.lower().split())

                if completion_words and target_words:
                    intersection = len(completion_words & target_words)
                    union = len(completion_words | target_words)
                    similarity = intersection / union if union > 0 else 0.0
                else:
                    similarity = 0.0

                # Quantum analysis
                quantum_coherence = self.quantum_analyzer.analyze_text(completion_text)

                # Combined evaluation score
                eval_score = (
                    self.config.perplexity_weight * similarity
                    + self.config.quantum_weight * quantum_coherence
                )

                eval_scores.append(eval_score)
                quantum_scores.append(quantum_coherence)

            except Exception as e:
                print(f"Evaluation error: {e}")
                continue

        # Log evaluation metrics
        if eval_scores:
            avg_eval_score = sum(eval_scores) / len(eval_scores)
            avg_quantum_score = sum(quantum_scores) / len(quantum_scores)

            self.eval_metrics.append(("eval/combined_score", avg_eval_score))
            self.eval_metrics.append(("eval/quantum_coherence", avg_quantum_score))
            self.eval_metrics.append(
                ("eval/perplexity_weight", self.config.perplexity_weight)
            )
            self.eval_metrics.append(
                ("eval/quantum_weight", self.config.quantum_weight)
            )

            print(
                f"Evaluation complete: avg_score={avg_eval_score:.3f}, avg_quantum={avg_quantum_score:.3f}"
            )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to Weights & Biases."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Calculate and add metrics from buffer
        for metric_name, values in self.metrics_buffer.items():
            if values:
                wandb_metrics[f"train/{metric_name}_avg"] = sum(values) / len(values)
                wandb_metrics[f"train/{metric_name}_max"] = max(values)
                wandb_metrics[f"train/{metric_name}_min"] = min(values)
                wandb_metrics[f"train/{metric_name}_std"] = np.std(values)

        # Add quantum-specific metrics
        wandb_metrics["quantum/n_qubits"] = self.config.n_qubits
        wandb_metrics["quantum/n_layers"] = self.config.n_layers
        wandb_metrics["quantum/weight"] = self.config.quantum_weight
        wandb_metrics["train/perplexity_weight"] = self.config.perplexity_weight

        # Clear the buffer
        self.metrics_buffer = {key: [] for key in self.metrics_buffer}

        # Add evaluation metrics
        for name, value in self.eval_metrics:
            wandb_metrics[name] = value

        self.eval_metrics = []

        # Log to wandb using the parent method
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    # Launch the environment with CLI arguments
    QuantumHybridEnv.cli()
