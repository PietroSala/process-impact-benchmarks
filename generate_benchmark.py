"""Generate a complexity grid counting both XOR and loop regions.

Run from the project directory: python generate_benchmark.py
Files are saved after each successful variant; rerun the command to resume.
The default sampler conditions operator weights on admissible expansions.
Use --sampling rejection to reproduce the earlier whole-process rejection.
"""

import argparse
import json
import math
from pathlib import Path
import random

from random_batch_generation import generate_process as generate_rejection_process
from constrained_generation import generate_constrained_process as generate_process
from random_diagram_generation import DEFAULT_PROBABILITIES, replace_random_underscore
from stats import max_independent_xor, max_nested_xor


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def load_existing(path, nested, independent, count):
    if not path.exists():
        return []
    processes = path.read_text(encoding="utf-8").splitlines()
    if len(processes) > count or len(set(processes)) != len(processes):
        raise ValueError(f"{path}: unexpected variant count or duplicate processes")
    for process in processes:
        try:
            if "!" not in process or "_" in process:
                raise ValueError("expected a complete process containing a loop")
            # With no placeholders this validates ancestry without changing it.
            replace_random_underscore(process)
            if (max_nested_xor(process), max_independent_xor(process)) != (nested, independent):
                raise ValueError("XOR + loop counts do not match the filename")
        except Exception as error:
            raise ValueError(f"{path}: invalid existing process: {error}") from error
    return processes


def generate_benchmark(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "benchmark_manifest.json"
    manifest = {
        "metrics": "xor_plus_loop_v1",
        "sampling": args.sampling,
        "sampler_version": 1,
        "probabilities_xor_parallel_sequence_loop": list(args.probabilities),
        "seed": args.seed,
        "max_nested": args.max_nested,
        "max_independent": args.max_independent,
        "processes_per_pair": args.processes,
        "replacements_per_attempt": args.replacements,
        "loop_nesting": "an XOR is required between consecutive ancestor loops",
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output settings differ from the saved benchmark; choose a new --output directory")
    elif any(output.glob("generated_processes_full_*.txt")):
        raise ValueError("Existing benchmark has no sampling metadata; choose a new --output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sampler = generate_process if args.sampling == "constrained" else generate_rejection_process
    total = args.max_nested * args.max_independent * args.processes
    completed = 0
    print(f"Target: {total} processes; metrics=XOR + loop; sampling={args.sampling}; seed={args.seed}; output={output}", flush=True)

    for nested in range(1, args.max_nested + 1):
        for independent in range(1, args.max_independent + 1):
            path = output / f"generated_processes_full_{nested}_{independent}.txt"
            processes = load_existing(path, nested, independent, args.processes)
            forbidden = set(processes)
            probabilities = args.probabilities
            completed += len(processes)
            if processes:
                print(f"XOR + loop ({nested}, {independent}): resuming from {len(processes)}/{args.processes}", flush=True)

            for variant in range(len(processes), args.processes):
                # A seed per variant also makes interrupted runs reproducible.
                random.seed(f"{args.seed}:{nested}:{independent}:{variant}")
                for trial in range(args.trials):
                    process = sampler(
                        probabilities, nested, independent,
                        args.replacements, forbidden,
                    )
                    if process is not None and "!" in process:
                        break
                    if (trial + 1) % 100 == 0:
                        print(
                            f"XOR + loop ({nested}, {independent}), variant {variant + 1}: "
                            f"searching ({trial + 1}/{args.trials} attempts)", flush=True,
                        )
                else:
                    raise RuntimeError(
                        f"XOR + loop ({nested}, {independent}): generated {len(processes)}/{args.processes} "
                        f"variants after {args.trials} attempts for the next variant. "
                        "Completed variants are saved. Increase --trials to resume, "
                        "or use a new --output when changing --replacements."
                    )

                processes.append(process)
                forbidden.add(process)
                temporary = path.with_suffix(".tmp")
                temporary.write_text("\n".join(processes) + "\n", encoding="utf-8")
                temporary.replace(path)
                completed += 1
                print(
                    f"[{completed}/{total}] XOR + loop ({nested}, {independent}): "
                    f"{len(processes)}/{args.processes} variants ({trial + 1} attempts)",
                    flush=True,
                )

    print(f"Complete: {completed} processes in {output.resolve()}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated_processes_loops_regenerated"))
    parser.add_argument("--max-nested", type=positive_integer, default=10, help="maximum nesting of XOR + loop regions")
    parser.add_argument("--max-independent", type=positive_integer, default=10, help="maximum independent XOR + loop regions")
    parser.add_argument("--processes", type=positive_integer, default=10, help="variants per XOR pair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=("constrained", "rejection"), default="constrained",
                        help="constrained: weighted admissible expansions; rejection: original whole-process rejection")
    parser.add_argument("--trials", type=positive_integer, default=10000, help="attempts per variant")
    parser.add_argument("--replacements", type=positive_integer, default=1000, help="expansions per attempt")
    parser.add_argument(
        "--probabilities", type=float, nargs=4, default=DEFAULT_PROBABILITIES,
        metavar=("XOR", "PARALLEL", "SEQUENCE", "LOOP"),
        help="fixed weights for all targets; default: 0.3 0.2 0.2 0.3",
    )
    args = parser.parse_args(argv)
    weights = args.probabilities
    if any(not math.isfinite(p) or p < 0 for p in weights) or abs(sum(weights) - 1) >= 1e-6:
        parser.error("probabilities must be finite, nonnegative, and sum to 1")
    if weights[0] == 0 or weights[3] == 0:
        parser.error("XOR and loop probabilities must be positive")
    if args.max_independent > 1 and weights[1] + weights[2] == 0:
        parser.error("independent XOR/loop regions require a positive parallel or sequence probability")
    try:
        generate_benchmark(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
