# NeuroSpec

**The Behavioral Compiler for Neural Networks.**

> CSS is to HTML as NeuroSpec is to neural networks.
> HTML defines structure; CSS declares style.
> Model weights define capability; NeuroSpec declares behavior.

---

## What is NeuroSpec?

NeuroSpec is a system that lets you **declare what you want a model to BE** and **compiles it to what the model DOES internally** — using sparse autoencoder (SAE) features, probes, and steering vectors.

Instead of:
- Prompt engineering (unreliable, fragile)
- RLHF (expensive, imprecise)
- Output filters (surface-level, bypassable)

NeuroSpec gives you:
- **Declarative behavioral specifications** in a purpose-built DSL
- **Feature-level compilation** to steering vectors and monitoring probes
- **Real-time enforcement** during inference via activation interception
- **Permanent training** via RLFR (Reinforcement Learning from Feature Rewards)

## Architecture

```
Human intent --> Behavioral Spec (.ns) --> Feature Constraints --> Model behavior
                                              |
                                    Compiled via SAE features
                                    Verified via probes
                                    Enforced via steering
                                    Permanent via RLFR
```

### The Five Layers

| Layer | Name | Function |
|-------|------|----------|
| 1 | **Microscope** | SAE-based feature catalog — decompose any model into searchable, labeled concepts |
| 2 | **Compiler** | DSL parser + compiler — translate `.ns` specs into feature-level interventions |
| 3 | **Runtime** | Inference engine with real-time steering and monitoring |
| 4 | **Immune System** | Continuous feature monitoring, anomaly detection, and automatic intervention |
| 5 | **Forge** | RLFR training — compile behavioral specs into permanent weight updates |

## Quick Start

### 1. Build a Feature Catalog

```bash
neurospec catalog build \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --sae goodfire/llama-3.3-70b-sae-l32 \
    --output catalog.json
```

### 2. Write a Behavioral Spec

```
# safe_coder.ns
model "meta-llama/Llama-3.3-70B-Instruct"
sae "goodfire/llama-3.3-70b-sae-l32"

spec safe_coder {
    suppress vulnerability_generation {
        features: ["sql_injection_pattern", "xss_vector", "path_traversal"]
        action: steer_away(strength=0.8)
    }

    require admit_uncertainty {
        when: api_knowledge_confidence < 0.5
        amplify: ["uncertainty_expression", "suggest_documentation"]
        strength: 0.6
    }

    monitor fabricated_api {
        features: ["invented_function_name", "hallucinated_parameter"]
        threshold: 0.3
        action: pause_and_retry(max_attempts=5)
    }
}
```

### 3. Compile

```bash
neurospec compile safe_coder.ns --catalog catalog.json --output safe_coder.compiled.json
```

### 4. Serve

```bash
neurospec serve --model meta-llama/Llama-3.3-70B-Instruct --spec safe_coder.compiled.json
```

### 5. (Optional) Train Permanently

```bash
neurospec train --model meta-llama/Llama-3.3-70B-Instruct --spec safe_coder.compiled.json --budget 500
```

## The Spec Language

NeuroSpec uses a purpose-built DSL (`.ns` files) for declaring model behavior:

### Keywords

| Keyword | Function | Compilation Target |
|---------|----------|--------------------|
| `suppress` | Reduce feature activation | Negative steering vector |
| `amplify` | Increase feature activation | Positive steering vector |
| `require` | Enforce a behavioral condition | Probe + conditional steering |
| `monitor` | Watch features in real-time | Runtime monitor with action |
| `alert_if` | Trigger alert on threshold | Alert callback |
| `compile_to_weights` | Make behavior permanent | RLFR training config |

### Composition

Specs can be composed and overridden:

```
import "safe_medical_assistant" as med
import "honest_coder" as coder

spec medical_coding_assistant = compose(med, coder) {
    override med.hallucination_risk.threshold: 0.2
}
```

## Installation

```bash
# Core only (DSL + compiler, no GPU needed)
pip install neurospec

# With runtime (requires PyTorch + GPU)
pip install neurospec[runtime]

# With monitoring dashboard
pip install neurospec[monitor]

# With RLFR training
pip install neurospec[forge]

# Everything
pip install neurospec[all]
```

## How It Works

### Feature Catalog (Layer 1)

NeuroSpec builds on **Sparse Autoencoders (SAEs)** — unsupervised models that decompose a neural network's internal activations into interpretable features. Each feature corresponds to a human-understandable concept (e.g., "medical uncertainty", "SQL injection pattern", "epistemic hedging").

The Feature Catalog automates the process of:
1. Running sample texts through the model
2. Collecting SAE activations per feature
3. Finding top-activating examples per feature
4. Using an LLM to label each feature with a name, description, and tags
5. Building a searchable index

### Compilation (Layer 2)

The compiler translates `.ns` specs into concrete interventions:

- **Feature resolution**: Maps natural-language feature names (e.g., `"sql_injection_pattern"`) to SAE feature indices by searching the catalog
- **Steering vector generation**: Creates vectors that amplify or suppress specific features during inference
- **Probe configuration**: Sets up lightweight classifiers for threshold-based monitoring
- **Conflict detection**: Validates that specs don't contain contradictory instructions

### Runtime (Layer 3)

The runtime wraps a HuggingFace model with PyTorch forward hooks that:
1. Intercept activations at the specified layer
2. Decode through the SAE into feature space
3. Apply steering vectors (amplify/suppress features)
4. Re-encode and replace the activations
5. Monitor feature values against thresholds
6. Execute actions when monitors trigger

### Immune System (Layer 4)

Continuous monitoring layer that:
- Tracks feature activation patterns in real-time
- Detects anomalies using z-score and trend analysis
- Triggers automatic interventions (steering, retry, kill)
- Logs all events for post-hoc audit

### Forge (Layer 5)

Based on Goodfire's [RLFR paper](https://www.goodfire.ai/research/rlfr), the Forge makes behavioral specs permanent:
1. Trains lightweight probes on the features specified in the spec
2. Uses probe outputs as reward signals for reinforcement learning
3. Probes run on a **frozen copy** of the base model (can't be hacked)
4. RL updates the model weights to permanently internalize the behavior

## Inspired By

This project draws on research from [Goodfire AI](https://www.goodfire.ai):

- **RLFR**: [Features as Rewards: Using Interpretability to Reduce Hallucinations](https://www.goodfire.ai/research/rlfr) — using probes as RL reward signals
- **Alzheimer's Biomarkers**: [Using Interpretability to Identify Novel Biomarkers](https://www.goodfire.ai/research/interpretability-for-alzheimers-detection) — extracting novel knowledge from model internals
- **Model Diff Amplification**: [Discovering Undesired Rare Behaviors](https://www.goodfire.ai/research/model-diff-amplification) — surfacing rare failure modes
- **R1 SAEs**: [Under the Hood of a Reasoning Model](https://www.goodfire.ai/research/under-the-hood-of-a-reasoning-model) — interpretability at frontier scale
- **Rakuten PII Detection**: [SAE Probes for PII Detection](https://www.goodfire.ai/research/rakuten-sae-probes-for-pii-detection) — production interpretability

## Project Status

This is an early-stage research project. The architecture is complete but individual components are at varying levels of maturity:

| Component | Status |
|-----------|--------|
| DSL Parser | Implemented |
| Compiler | Implemented |
| Feature Catalog | Implemented |
| Runtime Engine | Implemented |
| Monitoring | Implemented |
| RLFR Forge | Proof of concept |
| Marketplace | Proof of concept |
| CLI | Implemented |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).
