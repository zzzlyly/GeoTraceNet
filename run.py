from __future__ import annotations

import argparse
import copy

from emcad.config import load_config
from emcad.engine import configure_process_runtime, evaluate_run, smoke_run, train_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GeoTraceNet ISIC2018 runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", default="configs/927_isic2018.yaml")
        subparser.add_argument(
            "--override",
            action="append",
            default=[],
            help="Override a configuration value with key=value.",
        )

    train_parser = subparsers.add_parser("train", help="Train the model.")
    add_config_argument(train_parser)

    smoke_parser = subparsers.add_parser("smoke", help="Run a one-step smoke test.")
    add_config_argument(smoke_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    add_config_argument(eval_parser)
    eval_parser.add_argument("--run-id", default="")
    eval_parser.add_argument("--checkpoint", default="")
    eval_parser.add_argument("--split", default="test")

    prepare_parser = subparsers.add_parser("prepare", help="Prepare the configured dataset.")
    add_config_argument(prepare_parser)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_process_runtime()
    config = load_config(args.config, args.override)

    if args.command == "train":
        train_runs(config)
    elif args.command == "smoke":
        smoke_run(config)
    elif args.command == "eval":
        evaluate_run(
            config,
            run_id=args.run_id or None,
            checkpoint=args.checkpoint or None,
            split=args.split or None,
        )
    elif args.command == "prepare":
        from emcad.data.prepare import prepare_isic18_from_config

        prepare_isic18_from_config(copy.deepcopy(config))


if __name__ == "__main__":
    main()
