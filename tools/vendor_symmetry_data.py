"""Regenerate the vendored symmetry datasets under ``src/httk/atomistic/data/``.

Run via ``make symmetry-data DATA_GENERATORS=<path to data-generators checkout>``.

Four files are produced:

``symmetry_basics.json.gz``
    Copied byte-for-byte from the data-generators output. It is used as-is.

``spacegroup_setting_transforms.json.gz``
    A *derived subset*. The upstream ``transformations_hm_entry.json.gz`` is 5.2 MB
    compressed but 133 MB decompressed, of which httk needs only the per-setting
    ``hall_to_it_std_transform`` record — 112 KB, or 6 KB compressed. Shipping the whole
    file as a runtime dependency is not reasonable, so this script extracts that one
    field into a document of the same JSON-LD shape, carrying the source document's
    licence, creator, and provenance header forward unchanged so that the CC BY 4.0
    attribution chain is unbroken.

``spacegroup_subgroups.json.gz``
    A derived subset of ``transformations_std.json.gz`` containing only the per-IT-number
    Bärnighausen and continuous-normalizer sections.

``affine_normalizer_cosets.json.gz``
    Copied byte-for-byte from the data-generators output.

Nothing here runs at build or test time; the checked-in copies are authoritative.
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

VENDOR_DIR = Path(__file__).resolve().parent.parent / "src" / "httk" / "atomistic" / "data"

#: Header keys copied verbatim from the source document so attribution survives the slice.
PROVENANCE_KEYS = (
    "@context",
    "@type",
    "dcterms:created",
    "dcterms:issued",
    "version",
    "creator",
    "dcterms:license",
    "prov:wasGeneratedBy",
)


def build_transform_slice(source: Path) -> dict[str, Any]:
    """Extract ``hall_to_it_std_transform`` from ``transformations_hm_entry.json.gz``.

    The result keeps the source's JSON-LD envelope (context, licence, creator,
    provenance) and replaces only ``data``/``indicies``, so a reader can still see where
    the data came from and under what terms.
    """
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        document = json.load(handle)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in document["data"]["transformations_per_hm_entry"]:
        transform = entry.get("hall_to_it_std_transform")
        if transform is None:
            raise ValueError(f"record {entry.get('hall_entry')!r} has no hall_to_it_std_transform")
        hall_entry = transform["hall_entry"]
        if hall_entry in seen:
            continue
        seen.add(hall_entry)
        records.append(
            {
                "hall_entry": hall_entry,
                "it_number": transform["it_number"],
                "to_hall_entry": transform["to_hall_entry"],
                "affine_transformation": transform["affine_transformation"],
            }
        )

    records.sort(key=lambda record: (record["it_number"], record["hall_entry"]))
    index = {record["hall_entry"]: position for position, record in enumerate(records)}

    sliced = {key: document[key] for key in PROVENANCE_KEYS if key in document}
    sliced["@id"] = "urn:httk-atomistic:spacegroup_setting_transforms/0.1.0"
    sliced["dcterms:title"] = "Space-group setting transformations to the IT standard setting"
    sliced["rdfs:comment"] = (
        "The change-of-basis operation from each space-group setting's own coordinates to "
        "the International Tables standard (reference) setting, keyed on Hall entry. Each "
        "affine_transformation maps standard-setting coordinates into the setting's own "
        "coordinates as x_own = matrix @ x_std + vector. This dataset is a verbatim subset "
        "of the hall_to_it_std_transform records of the data-generators "
        "transformations_hm_entry dataset, extracted by "
        "httk-atomistic/tools/vendor_symmetry_data.py; no values are altered. The data in "
        "this file was generated using [cctbx](https://cctbx.github.io/)."
    )
    sliced["dcterms:source"] = document.get("@id")
    sliced["data"] = {"spacegroup_setting_transforms": records}
    sliced["indicies"] = {"index_hall_entry_to_spacegroup_setting_transforms": index}
    return sliced


def build_subgroup_slice(source: Path) -> dict[str, Any]:
    """Extract the subgroup and continuous-normalizer sections per IT number."""
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        document = json.load(handle)

    records = [
        {
            "it_number": entry["it_number"],
            "baernighausen": entry["baernighausen"],
            "continuous_normalizer": entry["continuous_normalizer"],
        }
        for entry in document["data"]["transformations_per_it_number"]
    ]
    records.sort(key=lambda record: record["it_number"])
    index = {str(record["it_number"]): position for position, record in enumerate(records)}

    sliced = {key: document[key] for key in PROVENANCE_KEYS if key in document}
    sliced["@id"] = "urn:httk-atomistic:spacegroup_subgroups/0.1.0"
    sliced["dcterms:title"] = "Space-group subgroup transformations and continuous normalizers"
    sliced["rdfs:comment"] = (
        "A verbatim per-IT-number subset of the transformations_std dataset, retaining only "
        "the baernighausen and continuous_normalizer sections. This dataset is extracted by "
        "httk-atomistic/tools/vendor_symmetry_data.py; no values are altered. Direction and "
        "convention documentation is intentionally deferred to the data README. The data in "
        "this file was generated using [cctbx](https://cctbx.github.io/)."
    )
    sliced["dcterms:source"] = document.get("@id")
    sliced["data"] = {"spacegroup_subgroups": records}
    sliced["indicies"] = {"index_it_number_to_spacegroup_subgroups": index}
    return sliced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_generators",
        type=Path,
        help="path to the data-generators checkout (the directory holding data/*.json.gz)",
    )
    args = parser.parse_args()

    source_dir = args.data_generators / "data"
    basics = source_dir / "symmetry_basics.json.gz"
    transforms = source_dir / "transformations_hm_entry.json.gz"
    subgroups = source_dir / "transformations_std.json.gz"
    cosets = source_dir / "affine_normalizer_cosets.json.gz"
    for path in (basics, transforms, subgroups, cosets):
        if not path.is_file():
            raise SystemExit(f"missing source dataset: {path}")

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(basics, VENDOR_DIR / "symmetry_basics.json.gz")
    print(f"copied {basics.name}")

    sliced = build_transform_slice(transforms)
    target = VENDOR_DIR / "spacegroup_setting_transforms.json.gz"
    # mtime=0 keeps the output byte-reproducible across runs.
    with gzip.GzipFile(target, "wb", mtime=0) as handle:
        handle.write(json.dumps(sliced, separators=(",", ":"), sort_keys=False).encode("utf-8"))
    count = len(sliced["data"]["spacegroup_setting_transforms"])
    print(f"wrote {target.name} ({count} settings, {target.stat().st_size} bytes)")

    sliced = build_subgroup_slice(subgroups)
    target = VENDOR_DIR / "spacegroup_subgroups.json.gz"
    with gzip.GzipFile(target, "wb", mtime=0) as handle:
        handle.write(json.dumps(sliced, separators=(",", ":"), sort_keys=False).encode("utf-8"))
    count = len(sliced["data"]["spacegroup_subgroups"])
    print(f"wrote {target.name} ({count} IT numbers, {target.stat().st_size} bytes)")

    target = VENDOR_DIR / "affine_normalizer_cosets.json.gz"
    shutil.copyfile(cosets, target)
    print(f"copied {cosets.name} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
