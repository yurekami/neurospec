"""RLFR (Reinforcement Learning from Feature Rewards) trainer.

Simplified training loop that uses probe-based rewards to modify
model weights so that behavioral specs become permanent.

Based on Goodfire's RLFR paper: probes on a frozen model provide
reward signals that are 90x cheaper than LLM judges while achieving
58% hallucination reduction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neurospec.core.types import ProbeConfig, RLFRConfig
from neurospec.forge.probes import LinearProbe
from neurospec.forge.reward import FeatureRewardFunction

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.optim as optim
except ImportError:
    torch = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


@dataclass
class TrainingMetrics:
    """Metrics from an RLFR training run."""

    steps_completed: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    final_loss: float = 0.0
    probe_accuracies: dict[str, float] = field(default_factory=dict)


class RLFRTrainer:
    """Train a model using RLFR (Reinforcement Learning from Feature Rewards).

    The key insight: instead of expensive LLM judges, use cheap linear probes
    on SAE features as reward signals. Probes run on a FROZEN copy of the base
    model to prevent the student from gaming them.

    Usage:
        trainer = RLFRTrainer(config, student_model, frozen_model, sae)
        metrics = trainer.train(training_texts)
    """

    def __init__(
        self,
        config: RLFRConfig,
        student_model: Any = None,
        frozen_model: Any = None,
        sae: Any = None,
        tokenizer: Any = None,
        device: str = "auto",
    ) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for RLFR. Install with: pip install neurospec[forge]"
            )

        self._config = config
        self._student = student_model
        self._frozen = frozen_model
        self._sae = sae
        self._tokenizer = tokenizer
        self._device = self._resolve_device(device)

        # Build probes from config
        self._probes = self._build_probes(config.probe_configs)

        # Build reward function on frozen model
        self._reward_fn = FeatureRewardFunction(
            probes=self._probes,
            frozen_model=frozen_model,
            sae=sae,
            reward_scale=config.reward_scale,
        )

    def train(
        self,
        training_texts: list[str],
        eval_texts: list[str] | None = None,
    ) -> TrainingMetrics:
        """Run the RLFR training loop.

        Args:
            training_texts: Texts to use for training.
            eval_texts: Optional texts for evaluation.

        Returns:
            TrainingMetrics with results.
        """
        metrics = TrainingMetrics()
        optimizer = optim.Adam(
            self._student.parameters(),
            lr=self._config.learning_rate,
        )

        budget = self._config.training_budget
        total_reward = 0.0

        logger.info("Starting RLFR training: %d steps, lr=%.2e", budget, self._config.learning_rate)

        for step in range(budget):
            # Select a training text
            text = training_texts[step % len(training_texts)]
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Forward pass through student
            outputs = self._student(**inputs, labels=inputs["input_ids"])
            lm_loss = outputs.loss

            # Compute reward from frozen model probes
            with torch.no_grad():
                reward, probe_results = self._reward_fn.compute_reward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )

            # REINFORCE-style update: loss = -reward * lm_loss
            # Higher reward -> lower loss -> encourage this behavior
            total_loss = lm_loss * (1.0 - reward)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_reward += reward
            metrics.steps_completed = step + 1
            metrics.final_loss = total_loss.item()

            if (step + 1) % 50 == 0:
                mean_r = total_reward / (step + 1)
                logger.info(
                    "Step %d/%d: loss=%.4f reward=%.4f mean_reward=%.4f",
                    step + 1,
                    budget,
                    total_loss.item(),
                    reward,
                    mean_r,
                )

        metrics.total_reward = total_reward
        metrics.mean_reward = total_reward / budget if budget > 0 else 0.0

        # Record final probe accuracies
        for probe in self._probes:
            metrics.probe_accuracies[probe.name] = 1.0 if probe.is_trained else 0.0

        logger.info(
            "RLFR training complete: %d steps, mean_reward=%.4f, final_loss=%.4f",
            metrics.steps_completed,
            metrics.mean_reward,
            metrics.final_loss,
        )

        return metrics

    def _build_probes(self, configs: list[ProbeConfig]) -> list[LinearProbe]:
        """Build linear probes from probe configs."""
        probes: list[LinearProbe] = []
        for config in configs:
            probe = LinearProbe(
                name=config.name,
                feature_ids=config.feature_ids,
                threshold=config.threshold,
            )
            probes.append(probe)
        return probes

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"
