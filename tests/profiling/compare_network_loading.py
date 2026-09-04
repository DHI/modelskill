"""Time `Network.from_epanet` across revisions, file sets and quantity filters.

Each case runs as its own subprocess of `profile_network_loading.py`, so no
module-level cache carries between them. A case can be pointed at a different
modelskill source tree with `--baseline-src`, which is how this compares the
current checkout against another revision:

    git worktree add ../modelskill-main main

    uv run python tests/profiling/compare_network_loading.py \\
        --res-path model.res --resx model.resx --inp model.inp \\
        --baseline-src ../modelskill-main/src --quantities Pressure

Both revisions then run in the one virtual environment, with only the
modelskill source differing - mikeio1d and networkx stay pinned to the same
builds, so the numbers reflect modelskill's own code and nothing else.

The `--baseline-src` tree need not know about the `quantities` filter: the
filtered cases only ever run against the current checkout.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "profile_network_loading.py"


@dataclass(frozen=True)
class Case:
    """One row of the comparison."""

    variant: str
    files: str
    src: Path | None
    companions: bool
    quantities: list[str] | None

    @property
    def label(self) -> str:
        return f"{self.variant}--{self.files}".replace(" ", "").replace("+", "-")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res-path", type=Path, required=True)
    parser.add_argument(
        "--resx",
        type=Path,
        default=None,
        help="Companion '.resx'. Given together with --inp, a second file set "
        "is timed alongside the res-only one.",
    )
    parser.add_argument("--inp", type=Path, default=None)
    parser.add_argument(
        "--baseline-src",
        type=Path,
        default=None,
        help="src/ directory of another modelskill checkout - typically a "
        "worktree of main - to time as the baseline. Omit to time only the "
        "current checkout.",
    )
    parser.add_argument(
        "--baseline-name",
        default="main",
        help="What to call the --baseline-src revision in the table.",
    )
    parser.add_argument(
        "--current-name",
        default="current",
        help="What to call the current checkout in the table, e.g. its branch name.",
    )
    parser.add_argument(
        "--quantities",
        nargs="+",
        default=None,
        help="Quantity name(s) for the filtered variant, e.g. Pressure. Omit "
        "to skip the filtered cases.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="How many times to run the whole case list. Each measurement is "
        "one load in a fresh process - loading this much data repeatedly in "
        "one process builds memory pressure that swamps the differences being "
        "measured. The reported figure is the minimum over the rounds.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Also dump a cProfile pass per case, for drilling into a "
        "surprising number afterwards. Roughly doubles the runtime.",
    )
    return parser.parse_args()


def _build_cases(args: argparse.Namespace) -> list[Case]:
    companions = [n for n, p in (("resx", args.resx), ("inp", args.inp)) if p]
    file_sets = [("res", False)]
    if companions:
        file_sets.append(("res+" + "+".join(companions), True))

    variants: list[tuple[str, Path | None, list[str] | None]] = []
    if args.baseline_src is not None:
        variants.append((args.baseline_name, args.baseline_src.resolve(), None))
    variants.append((args.current_name, None, None))
    if args.quantities:
        filtered = f"{args.current_name} +{','.join(args.quantities)}"
        variants.append((filtered, None, args.quantities))

    return [
        Case(variant, files, src, companions, quantities)
        for variant, src, quantities in variants
        for files, companions in file_sets
    ]


def _run_once(case: Case, args: argparse.Namespace, *, profile: bool) -> dict | None:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--res-path",
        str(args.res_path),
        # One load, no in-process warm-up. Every measurement therefore includes
        # the same one-off mikeio1d start-up cost, which is what a user pays
        # too, and no measurement is distorted by an earlier load's memory.
        "--repeat",
        "1",
        "--warmup",
        "0",
        "--label",
        case.label,
        "--json",
    ]
    if case.companions:
        if args.resx is not None:
            cmd += ["--resx", str(args.resx)]
        if args.inp is not None:
            cmd += ["--inp", str(args.inp)]
    if case.quantities:
        cmd += ["--quantities", *case.quantities]
    if profile:
        cmd += ["--profile"]

    env = dict(os.environ)
    if case.src is None:
        env.pop("PYTHONPATH", None)
    else:
        # Wins over the .pth entry that site-packages adds for the editable
        # install, so this subprocess imports the baseline tree instead.
        env["PYTHONPATH"] = str(case.src)

    print(f"  {case.variant:<28} {case.files:<14} ... ", end="", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print("FAILED")
        print(proc.stdout, proc.stderr, sep="\n", file=sys.stderr)
        return None

    result = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"{result['min']:.2f}s")
    return {**result, "variant": case.variant, "files": case.files}


def _table(results: list[dict], baseline_name: str) -> str:
    """CPU time leads, because it is the column the machine cannot distort."""
    by_files = {r["files"]: r for r in results if r["variant"] == baseline_name}

    lines = [
        "| revision | files | cpu min | cpu speedup | wall min | wall speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        ref = by_files.get(r["files"])
        cpu_up = "-" if ref is None else f"{ref['cpu_min'] / r['cpu_min']:.2f}x"
        wall_up = "-" if ref is None else f"{ref['min'] / r['min']:.2f}x"
        lines.append(
            f"| {r['variant']} | {r['files']} | {r['cpu_min']:.2f}s | {cpu_up} | "
            f"{r['min']:.2f}s | {wall_up} |"
        )
    return "\n".join(lines)


def _rounds(results: list[dict]) -> str:
    lines = [
        "| revision | files | cpu per round | wall per round |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        cpu = ", ".join(f"{s:.1f}" for s in r["cpu_samples"])
        wall = ", ".join(f"{s:.1f}" for s in r["samples"])
        lines.append(f"| {r['variant']} | {r['files']} | {cpu} | {wall} |")
    return "\n".join(lines)


def _sanity(results: list[dict]) -> str:
    """Different node or reach counts would mean the cases are not comparable."""
    lines = [
        "| revision | files | nodes | reaches | quantities loaded | source |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['files']} | {r['n_nodes']} | {r['n_reaches']} | "
            f"{', '.join(r['quantities'])} | {r['modelskill']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    cases = _build_cases(args)

    print(f"Timing {len(cases)} cases over {args.rounds} rounds.")

    samples: dict[str, list[float]] = {c.label: [] for c in cases}
    cpu_samples: dict[str, list[float]] = {c.label: [] for c in cases}
    latest: dict[str, dict] = {}
    for round_ in range(args.rounds):
        print(f"\nRound {round_ + 1}/{args.rounds}")
        # Reversed on alternate rounds so a machine that drifts over the run
        # biases the first and last case equally rather than only the last.
        ordered = cases if round_ % 2 == 0 else list(reversed(cases))
        for case in ordered:
            profile = args.profile and round_ == args.rounds - 1
            result = _run_once(case, args, profile=profile)
            if result is None:
                continue
            samples[case.label].append(result["seconds"][0])
            cpu_samples[case.label].append(result["cpu_seconds"][0])
            latest[case.label] = result

    results = [
        {
            **latest[c.label],
            "samples": samples[c.label],
            "cpu_samples": cpu_samples[c.label],
            "min": min(samples[c.label]),
            "median": statistics.median(samples[c.label]),
            "cpu_min": min(cpu_samples[c.label]),
        }
        for c in cases
        if samples[c.label]
    ]
    if not results:
        sys.exit("Every case failed.")

    print("\n" + _table(results, args.baseline_name))
    print("\n" + _rounds(results))
    print("\n" + _sanity(results))

    counts = {(r["n_nodes"], r["n_reaches"]) for r in results}
    if len(counts) > 1:
        print(
            "\nWARNING: the cases loaded different networks "
            f"({counts}), so the times are not comparable."
        )

    # A wide spread within a case means the machine was noisy and the
    # between-case differences above deserve less trust.
    for name, key in (("cpu", "cpu_samples"), ("wall", "samples")):
        spread = [max(r[key]) / min(r[key]) for r in results]
        print(
            f"\nWithin-case {name} spread (max/min): "
            f"{statistics.mean(spread):.2f}x mean, {max(spread):.2f}x worst. "
            "1.00x is perfectly repeatable."
        )


if __name__ == "__main__":
    main()
