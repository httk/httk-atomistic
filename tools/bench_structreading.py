"""Benchmark serial loading of the structreading CIF corpus."""

import argparse
import cProfile
import io
import pstats
import statistics
import time
from pathlib import Path

from httk.core import load


def _read(path: Path) -> bool:
    """Read one CIF, returning whether repair was needed."""
    try:
        load(str(path))
    except ValueError:
        load(str(path), repair=True)
        return True
    return False


def _pass(paths: list[Path]) -> tuple[list[tuple[Path, float, bool]], float]:
    start = time.perf_counter()
    results = []
    for path in paths:
        file_start = time.perf_counter()
        fallback = _read(path)
        results.append((path, time.perf_counter() - file_start, fallback))
    return results, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true", help="profile the pass with cProfile")
    args = parser.parse_args()
    paths = sorted(Path(__file__).parent.parent.joinpath("tests/fixtures/structreading").glob("*.cif"))
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        results, total = _pass(paths)
        profiler.disable()
    else:
        results, total = _pass(paths)

    for path, elapsed, fallback in results:
        print(f"{path.name}: {elapsed:.6f}s" + (" (repair)" if fallback else ""))
    times = [elapsed for _, elapsed, _ in results]
    print(f"total: {total:.6f}s")
    print(f"mean: {statistics.mean(times):.6f}s")
    print(f"median: {statistics.median(times):.6f}s")
    print(f"repair fallbacks: {sum(fallback for _, _, fallback in results)}")
    if args.profile:
        output = io.StringIO()
        pstats.Stats(profiler, stream=output).sort_stats("tottime").print_stats(25)
        print(output.getvalue(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
