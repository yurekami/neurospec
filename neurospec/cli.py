"""NeuroSpec CLI — The Behavioral Compiler for Neural Networks.

Usage:
    neurospec compile <spec_file> [--catalog <catalog_path>] [--output <output_path>]
    neurospec serve --model <model_id> --spec <compiled_spec> [--port <port>]
    neurospec catalog build --model <model_id> --sae <sae_path> [--texts <texts_path>] [--output <output_path>]
    neurospec catalog search <query> [--catalog <catalog_path>]
    neurospec monitor --model <model_id> --spec <compiled_spec>
    neurospec train --spec <compiled_spec> --model <model_id> [--budget <steps>]
    neurospec registry list
    neurospec registry publish <spec_file>
    neurospec registry install <spec_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurospec",
        description="NeuroSpec: The Behavioral Compiler for Neural Networks",
        epilog="Declare what you want a model to BE. Compile it to what it DOES internally.",
    )
    parser.add_argument("--version", action="version", version="neurospec 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- compile ---
    compile_parser = subparsers.add_parser("compile", help="Compile a .ns spec file")
    compile_parser.add_argument("spec_file", type=str, help="Path to .ns spec file")
    compile_parser.add_argument(
        "--catalog", type=str, default=None, help="Path to feature catalog JSON"
    )
    compile_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path for compiled spec"
    )

    # --- serve ---
    serve_parser = subparsers.add_parser(
        "serve", help="Serve a model with compiled spec enforcement"
    )
    serve_parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    serve_parser.add_argument("--spec", type=str, required=True, help="Path to compiled spec")
    serve_parser.add_argument("--sae", type=str, default=None, help="SAE path")
    serve_parser.add_argument("--port", type=int, default=8420, help="Server port")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")

    # --- catalog ---
    catalog_parser = subparsers.add_parser("catalog", help="Feature catalog operations")
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_command")

    build_parser = catalog_sub.add_parser("build", help="Build a feature catalog")
    build_parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    build_parser.add_argument("--sae", type=str, required=True, help="SAE path")
    build_parser.add_argument(
        "--texts", type=str, default=None, help="Path to text dataset (JSONL)"
    )
    build_parser.add_argument(
        "--output", "-o", type=str, default="catalog.json", help="Output catalog path"
    )
    build_parser.add_argument("--layer", type=int, default=None, help="Model layer to extract from")
    build_parser.add_argument(
        "--top-k", type=int, default=20, help="Top-k activating examples per feature"
    )
    build_parser.add_argument(
        "--labeler", type=str, default="openai", help="LLM provider for labeling"
    )

    search_parser = catalog_sub.add_parser("search", help="Search feature catalog")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--catalog", type=str, default="catalog.json", help="Catalog path")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")

    # --- monitor ---
    monitor_parser = subparsers.add_parser("monitor", help="Real-time feature monitoring dashboard")
    monitor_parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    monitor_parser.add_argument("--spec", type=str, required=True, help="Compiled spec path")
    monitor_parser.add_argument("--sae", type=str, default=None, help="SAE path")

    # --- train ---
    train_parser = subparsers.add_parser(
        "train", help="RLFR training — compile spec to permanent weights"
    )
    train_parser.add_argument("--spec", type=str, required=True, help="Compiled spec path")
    train_parser.add_argument("--model", type=str, required=True, help="Model ID or path")
    train_parser.add_argument("--budget", type=int, default=500, help="Training steps budget")
    train_parser.add_argument("--output", "-o", type=str, default=None, help="Output model path")

    # --- registry ---
    registry_parser = subparsers.add_parser("registry", help="Spec marketplace operations")
    registry_sub = registry_parser.add_subparsers(dest="registry_command")

    registry_sub.add_parser("list", help="List available specs")

    pub_parser = registry_sub.add_parser("publish", help="Publish a spec")
    pub_parser.add_argument("spec_file", type=str, help="Spec file to publish")

    install_parser = registry_sub.add_parser("install", help="Install a spec")
    install_parser.add_argument("spec_id", type=str, help="Spec ID to install")

    return parser


def cmd_compile(args: argparse.Namespace) -> int:
    """Compile a .ns spec file into steering vectors and monitoring config."""
    from neurospec.dsl.lexer import Lexer
    from neurospec.dsl.parser import Parser
    from neurospec.dsl.validator import validate_spec
    from neurospec.compiler.compiler import NeuroSpecCompiler
    from neurospec.compiler.serializer import serialize_to_json
    from neurospec.core.types import FeatureCatalog

    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(f"Error: Spec file not found: {spec_path}", file=sys.stderr)
        return 1

    source = spec_path.read_text(encoding="utf-8")
    print(f"Parsing {spec_path.name}...")

    # Lex
    lexer = Lexer(source)
    tokens = lexer.tokenize()

    # Parse
    parser = Parser(tokens)
    spec_file = parser.parse()

    # Validate
    errors = validate_spec(spec_file)
    if any(e.severity == "error" for e in errors):
        for e in errors:
            print(f"  [{e.severity}] line {e.line}: {e.message}", file=sys.stderr)
        return 1
    for e in errors:
        print(f"  [warning] line {e.line}: {e.message}")

    # Load catalog if provided
    catalog = None
    if args.catalog:
        catalog_path = Path(args.catalog)
        if catalog_path.exists():
            catalog = FeatureCatalog.load(str(catalog_path))
            print(f"Loaded catalog with {len(catalog.features)} features")
        else:
            print(f"Warning: Catalog not found at {catalog_path}, compiling without resolution")

    # Compile
    compiler = NeuroSpecCompiler(catalog=catalog)
    for spec_decl in spec_file.specs:
        result = compiler.compile(spec_decl)
        print(f"Compiled spec '{spec_decl.name}':")
        print(f"  Steering vectors: {len(result.steering_vectors)}")
        print(f"  Probes: {len(result.probes)}")
        print(f"  Monitors: {len(result.monitors)}")

        # Write output
        output_path = args.output or f"{spec_decl.name}.compiled.json"
        Path(output_path).write_text(serialize_to_json(result), encoding="utf-8")
        print(f"  Written to: {output_path}")

    print("Compilation complete.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve a model with compiled spec enforcement."""
    from neurospec.runtime.engine import NeuroSpecEngine
    from neurospec.runtime.serving import start_server

    print(f"Loading model: {args.model}")
    engine = NeuroSpecEngine(model_id=args.model, sae_path=args.sae or "", device="auto")

    print(f"Loading spec: {args.spec}")
    engine.load_spec(args.spec)

    print(f"Starting server on {args.host}:{args.port}")
    start_server(engine, host=args.host, port=args.port)
    return 0


def cmd_catalog_build(args: argparse.Namespace) -> int:
    """Build a feature catalog for a model + SAE."""
    from neurospec.catalog.builder import CatalogBuilder

    print(f"Building catalog for {args.model} with SAE {args.sae}")
    builder = CatalogBuilder(
        model_id=args.model,
        sae_path=args.sae,
        labeler_provider=args.labeler,
    )

    texts = None
    if args.texts:
        texts_path = Path(args.texts)
        if texts_path.exists():
            texts = [
                json.loads(line)["text"]
                for line in texts_path.read_text(encoding="utf-8").strip().split("\n")
            ]
            print(f"Loaded {len(texts)} texts from {texts_path}")

    catalog = builder.build_catalog(
        texts=texts,
        layer=args.layer,
        top_k=args.top_k,
    )

    catalog.save(args.output)
    print(f"Catalog saved to {args.output} ({len(catalog.features)} features)")
    return 0


def cmd_catalog_search(args: argparse.Namespace) -> int:
    """Search a feature catalog."""
    from neurospec.core.types import FeatureCatalog

    catalog = FeatureCatalog.load(args.catalog)
    results = catalog.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching features found.")
        return 0

    print(f"Found {len(results)} features matching '{args.query}':\n")
    for feature in results:
        tags = ", ".join(feature.tags) if feature.tags else "none"
        print(f"  [{feature.id:>6}] {feature.label}")
        print(f"          {feature.description}")
        print(f"          tags: {tags} | confidence: {feature.confidence:.2f}")
        print()

    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Launch real-time feature monitoring dashboard."""
    from neurospec.monitor.dashboard import MonitorDashboard

    print(f"Starting monitor for {args.model} with spec {args.spec}")
    print("(Dashboard would launch here — requires model + SAE loaded)")
    # In full implementation: load model, SAE, spec, start Guardian + Dashboard
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run RLFR training to compile spec into permanent weights."""
    print(f"RLFR training: {args.model} with spec {args.spec}")
    print(f"Budget: {args.budget} steps")
    print("(Training would start here — requires GPU + model loaded)")
    # In full implementation: load model, build probes from spec, run RLFRTrainer
    return 0


def cmd_registry_list(args: argparse.Namespace) -> int:
    """List specs in the marketplace registry."""
    from neurospec.marketplace.registry import SpecRegistry

    registry = SpecRegistry()
    specs = registry.list_specs()

    if not specs:
        print("No specs in registry. Publish one with: neurospec registry publish <spec.ns>")
        return 0

    for spec_meta in specs:
        print(f"  {spec_meta.name} v{spec_meta.version} by {spec_meta.author}")
        print(f"    {spec_meta.description}")
        print(f"    model: {spec_meta.model_id} | downloads: {spec_meta.downloads}")
        print()

    return 0


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "compile": cmd_compile,
        "serve": cmd_serve,
        "monitor": cmd_monitor,
        "train": cmd_train,
    }

    if args.command == "catalog":
        if args.catalog_command == "build":
            return cmd_catalog_build(args)
        elif args.catalog_command == "search":
            return cmd_catalog_search(args)
        else:
            parser.parse_args(["catalog", "--help"])
            return 0
    elif args.command == "registry":
        if args.registry_command == "list":
            return cmd_registry_list(args)
        else:
            print(f"Registry command '{args.registry_command}' not yet implemented")
            return 0
    elif args.command in dispatch:
        return dispatch[args.command](args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
