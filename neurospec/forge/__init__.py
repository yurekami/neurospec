"""NeuroSpec forge — RLFR training to compile specs into permanent weights.

Usage:
    from neurospec.forge import RLFRTrainer, LinearProbe

    trainer = RLFRTrainer(config, student_model, frozen_model, sae, tokenizer)
    metrics = trainer.train(training_texts)
"""

from neurospec.forge.probes import LinearProbe, ProbeResult
from neurospec.forge.reward import FeatureRewardFunction
from neurospec.forge.trainer import RLFRTrainer, TrainingMetrics

__all__ = [
    "FeatureRewardFunction",
    "LinearProbe",
    "ProbeResult",
    "RLFRTrainer",
    "TrainingMetrics",
]
