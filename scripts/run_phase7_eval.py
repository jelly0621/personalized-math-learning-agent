"""Command-line entry point for the isolated Phase 7 pilot evaluation."""

import argparse

from evals.pilot_runner import run_phase7_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 7 pilot evaluation.")
    parser.add_argument("--suite", default="pilot", choices=["pilot"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=50)
    parser.add_argument("--max-focuses", type=int, default=3)
    parser.add_argument("--challenge-cases", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_phase7_eval(
        dry_run=args.dry_run,
        suite=args.suite,
        max_cases=args.max_cases,
        max_llm_calls=args.max_llm_calls,
        max_focuses=args.max_focuses,
        challenge_cases=args.challenge_cases,
    )
    print(f"Phase 7 artifacts: {artifacts.output_directory}")
    print(f"LLM calls: {artifacts.summary.total_llm_calls}")
    print(f"Stopped by budget: {artifacts.summary.stopped_by_budget}")


if __name__ == "__main__":
    main()
