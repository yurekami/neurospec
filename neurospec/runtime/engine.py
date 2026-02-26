"""NeuroSpec inference engine — loads models, applies specs, runs generation.

The central runtime component that:
1. Loads a model and SAE
2. Loads a compiled spec
3. Registers forward hooks for steering and monitoring
4. Runs inference with real-time behavioral enforcement
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neurospec.core.types import (
    ActionKind,
    MonitorConfig,
    SpecCompilationResult,
    SteeringDirection,
)
from neurospec.runtime.actions import ActionExecutor
from neurospec.runtime.hooks import ActivationHook, HookState, SteeringInstruction

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class NeuroSpecEngine:
    """Runtime engine for NeuroSpec behavioral enforcement.

    Usage:
        engine = NeuroSpecEngine(model_id="meta-llama/...", sae_path="...")
        engine.load_spec("spec.compiled.json")
        result = engine.generate("Tell me about security")
    """

    def __init__(
        self,
        model_id: str,
        sae_path: str,
        device: str = "auto",
        layer: int | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required. Install with: pip install neurospec[runtime]")

        self._model_id = model_id
        self._sae_path = sae_path
        self._device = self._resolve_device(device)
        self._layer = layer

        self._model: Any = None
        self._tokenizer: Any = None
        self._sae: Any = None
        self._hook: ActivationHook | None = None
        self._hook_state = HookState()
        self._action_executor = ActionExecutor(engine=self)
        self._spec: SpecCompilationResult | None = None
        self._monitors: list[MonitorConfig] = []

        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def spec(self) -> SpecCompilationResult | None:
        return self._spec

    def load(self) -> None:
        """Load the model, tokenizer, and SAE."""
        if self._loaded:
            return

        logger.info("Loading model: %s", self._model_id)
        self._model, self._tokenizer = self._load_model()

        logger.info("Loading SAE: %s", self._sae_path)
        self._sae = self._load_sae()

        self._loaded = True
        logger.info("Engine loaded successfully")

    def load_spec(self, spec_path: str) -> None:
        """Load a compiled spec and configure steering + monitoring.

        Args:
            spec_path: Path to a compiled spec JSON file.
        """
        spec_data = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self._spec = SpecCompilationResult.from_dict(spec_data)

        # Convert steering vectors to instructions
        instructions: list[SteeringInstruction] = []
        for sv in self._spec.steering_vectors:
            multiplier = sv.strength if sv.direction == SteeringDirection.AMPLIFY else -sv.strength
            instructions.append(
                SteeringInstruction(
                    feature_ids=sv.feature_ids,
                    multiplier=multiplier,
                    label=sv.label,
                )
            )

        self._hook_state.steering_instructions = instructions
        self._monitors = self._spec.monitors

        logger.info(
            "Loaded spec '%s': %d steering vectors, %d monitors",
            self._spec.spec_name,
            len(self._spec.steering_vectors),
            len(self._monitors),
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text with behavioral enforcement.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.

        Returns:
            Generated text string.
        """
        if not self._loaded:
            self.load()

        # Register hooks
        self._register_hooks()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self._run_generation(prompt, max_new_tokens, temperature, top_p)

                # Check monitors
                should_retry = self._check_monitors()
                if should_retry and attempt < max_retries - 1:
                    logger.info("Monitor triggered retry (attempt %d/%d)", attempt + 1, max_retries)
                    continue

                return result

            except _GenerationKilled as exc:
                logger.warning("Generation killed: %s", exc)
                return f"[Generation halted: {exc}]"
            finally:
                if attempt == max_retries - 1:
                    self._remove_hooks()

        return result

    def _run_generation(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Run the actual generation loop."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _check_monitors(self) -> bool:
        """Check all monitors against current feature state.

        Returns True if a retry is needed.
        """
        if self._hook_state.last_features is None:
            return False

        features = self._hook_state.last_features
        should_retry = False

        for monitor in self._monitors:
            for feat_id in monitor.feature_ids:
                if feat_id < features.shape[-1]:
                    max_activation = float(features[..., feat_id].max())
                    if max_activation > monitor.threshold:
                        logger.warning(
                            "Monitor '%s' triggered: feature %d activation %.3f > threshold %.3f",
                            monitor.name,
                            feat_id,
                            max_activation,
                            monitor.threshold,
                        )
                        result = self._action_executor.execute(
                            monitor.action,
                            params=monitor.action_params,
                            context={"monitor_name": monitor.name, "feature_id": feat_id},
                        )
                        if result.should_retry:
                            should_retry = True
                        if result.should_kill:
                            raise _GenerationKilled(result.message)

        return should_retry

    def _register_hooks(self) -> None:
        """Register activation hooks on the target layer."""
        if self._hook is not None:
            return

        if self._sae is None:
            return

        target = self._get_target_layer()
        self._hook = ActivationHook(sae=self._sae, state=self._hook_state)
        self._hook.register(target)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _get_target_layer(self) -> Any:
        """Get the target layer for hook registration."""
        layers = self._get_model_layers()
        if self._layer is not None and self._layer < len(layers):
            return layers[self._layer]
        # Default to middle layer
        return layers[len(layers) // 2]

    def _get_model_layers(self) -> list[Any]:
        """Extract transformer layers from the model."""
        if hasattr(self._model, "model"):
            inner = self._model.model
            if hasattr(inner, "layers"):
                return list(inner.layers)
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                return list(inner.decoder.layers)
        if hasattr(self._model, "transformer"):
            transformer = self._model.transformer
            if hasattr(transformer, "h"):
                return list(transformer.h)
        raise ValueError("Could not find transformer layers in model")

    def _load_model(self) -> tuple[Any, Any]:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            device_map="auto" if self._device == "auto" else None,
        )
        if self._device not in ("auto",):
            model = model.to(self._device)
        model.eval()
        return model, tokenizer

    def _load_sae(self) -> Any:
        """Load the SAE."""
        from neurospec.runtime.sae_wrapper import SAEWrapper

        return SAEWrapper.from_path(self._sae_path, device=self._device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"


class _GenerationKilled(Exception):
    """Internal exception raised when a monitor kills generation."""
