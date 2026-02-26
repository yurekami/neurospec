# Contributing to NeuroSpec

## Development Setup

```bash
git clone https://github.com/yurekami/neurospec.git
cd neurospec
pip install -e ".[dev,all]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

- Python 3.10+
- Use `ruff` for linting: `ruff check .`
- Use `mypy` for type checking: `mypy neurospec/`
- Prefer dataclasses over pydantic for simple types
- Use Protocol over ABC for interfaces
- Conditional imports for optional dependencies (torch, transformers, etc.)

## Project Structure

```
neurospec/
    core/           # Shared types, config
    catalog/        # Feature extraction and labeling (Phase 0)
    dsl/            # DSL lexer, parser, AST (Phase 1)
    compiler/       # Spec compilation to interventions (Phase 1)
    runtime/        # Inference engine with steering (Phase 2)
    monitor/        # Real-time feature monitoring (Phase 3)
    forge/          # RLFR training (Phase 4)
    marketplace/    # Spec sharing and composition (Phase 5)
    cli.py          # Command-line interface
specs/
    examples/       # Example .ns spec files
tests/
    test_catalog/
    test_compiler/
    test_runtime/
    test_monitor/
    test_forge/
```

## Adding a New Feature

1. Write the spec in a `.ns` file first (dogfooding)
2. Add types to `core/types.py` if needed
3. Implement in the appropriate module
4. Add tests
5. Update CLI if it exposes a new command

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
