"""Generate the exact CIF structure-reading regression golden."""

import argparse
import gzip
import io
import json
from pathlib import Path

from httk.atomistic._structreading import structreading_golden


def main() -> int:
    """Generate a compressed golden from a directory of CIF files.

    :return: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        default=Path("tests/fixtures/structreading"),
        help="directory holding CIF files (default: tests/fixtures/structreading)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/structreading_golden.json.gz"),
        help="compressed JSON golden to write",
    )
    args = parser.parse_args()
    files = sorted(path for path in args.corpus.glob("*.cif") if not path.name.startswith("."))
    if not files:
        parser.error(f"no CIF files in {args.corpus}")
    golden = {path.name: structreading_golden(path) for path in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle,
    ):
        json.dump(golden, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
