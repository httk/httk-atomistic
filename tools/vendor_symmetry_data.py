"""Copy the six canonical symmetry datasets into the package data directory.

Run via ``make symmetry-data DATA_GENERATORS=<path to data-generators checkout>``.
The canonical artifact is vendored in whichever published format is appropriately sized;
at current sizes that is ``.json.gz`` for all six. The upstream ``.sqlar`` twins exist
for large-dataset/lazy-access use. These six canonical ``.json.gz`` files are copied
byte-for-byte from the upstream data-generators ``data/`` directory. The canonical
per-concern split happened upstream; this replaces the old vendored slices, including the
two derived documents that httk previously generated locally.

Nothing here runs at build or test time; the checked-in copies are authoritative.
"""

import argparse
import shutil
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent.parent / "src" / "httk" / "atomistic" / "data"
DATASETS = (
    "symmetry_basics.json.gz",
    "spacegroup_setting_transforms.json.gz",
    "baernighausen_std.json.gz",
    "continuous_euclidean_normalizer_std.json.gz",
    "affine_normalizer_cosets.json.gz",
    "isomorphic_subgroups_std.json.gz",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_generators",
        type=Path,
        help="path to the data-generators checkout (the directory holding data/*.json.gz)",
    )
    args = parser.parse_args()

    source_dir = args.data_generators / "data"
    sources = [source_dir / name for name in DATASETS]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(f"missing source dataset(s): {missing_text}")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = VENDOR_DIR / source.name
        shutil.copyfile(source, target)
        print(f"copied {source.name} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
