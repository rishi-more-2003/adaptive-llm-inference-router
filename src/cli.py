from __future__ import annotations

import argparse
import json

from src.router.router import AdaptiveInferenceRouter
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive LLM Inference Router CLI.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-route", help="Inspect the route for a prompt.")
    inspect.add_argument("--prompt", required=True)
    inspect.add_argument("--max-tokens", type=int, default=128)
    inspect.add_argument("--latency-target-ms", type=int, default=None)
    inspect.add_argument("--batch-size", type=int, default=1)
    inspect.add_argument(
        "--update-prefix-cache",
        action="store_true",
        help="Update the in-memory prefix cache before deciding.",
    )

    args = parser.parse_args()
    if args.command == "inspect-route":
        config = load_config(args.config)
        router = AdaptiveInferenceRouter(config)
        decision = router.inspect_route(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            latency_target_ms=args.latency_target_ms,
            batch_size=args.batch_size,
            update_prefix_cache=args.update_prefix_cache,
        )
        print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
