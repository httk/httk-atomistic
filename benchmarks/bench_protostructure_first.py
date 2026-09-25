"""Compare baseline and protostructure-first ASU canonicalizers on raw CIF files.

Run from the atomistic checkout, for example::

    python benchmarks/bench_protostructure_first.py \
        --corpus-root ../../../databases/DATA --output /tmp/protostructure-first.jsonl

An optional CSV or JSONL manifest has ``corpus,path`` fields and an optional
``sha256``. Paths are absolute or relative to ``--corpus-root``. Corpus names
are ``cod``, ``icsd-standardized`` and ``icsd-experimental``; automatic discovery
samples COD and ICSD standardized files.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

_CORPORA = ("cod", "icsd-standardized", "icsd-experimental")
_DEFAULT_CORPORA = _CORPORA[:2]
_METHODS = ("old", "new")


class _Deadline(BaseException):
    """Escape a canonicalization call when its deadline expires."""


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _Deadline


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    if not methods or len(set(methods)) != len(methods) or any(item not in _METHODS for item in methods):
        raise argparse.ArgumentTypeError("use one or both of: old,new")
    return methods


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    versions = {}
    for distribution in ("httk-atomistic", "httk-core", "spglib"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    checkout = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": versions,
        "atomistic_git_head": head.stdout.strip() if head.returncode == 0 else None,
    }


def _discover(root: Path, manifest: Path | None) -> list[dict[str, str]]:
    if manifest is not None:
        rows = []
        if manifest.suffix == ".jsonl":
            source_rows = [
                row
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
                for row in (json.loads(line),)
                if row.get("record") not in ("selection", "benchmark", "source")
            ]
        else:
            with manifest.open(newline="", encoding="utf-8") as source:
                source_rows = list(csv.DictReader(source))
        for row in source_rows:
            corpus = row.get("corpus", "").strip().lower()
            if corpus not in _CORPORA:
                raise ValueError(f"manifest corpus must be one of {_CORPORA}: {corpus!r}")
            raw_path = Path(row.get("path", ""))
            path = raw_path if raw_path.is_absolute() else root / raw_path
            rows.append({"corpus": corpus, "path": str(path.resolve()), "sha256": row.get("sha256", "").strip()})
        if not rows:
            raise ValueError(f"manifest has no data rows: {manifest}")
        return rows

    locations = {
        "cod": root / "COD" / "cif",
        "icsd-standardized": root / "ICSD" / "2026.1" / "cif-standardized",
    }
    rows = []
    for corpus, directory in locations.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {corpus} CIF directory: {directory}")
        rows.extend({"corpus": corpus, "path": str(path), "sha256": ""} for path in sorted(directory.rglob("*.cif")))
    return rows


def _load(path: Path, corpus: str) -> Any:
    from httk.core import load

    from httk.atomistic import ASUStructureView

    if corpus == "cod":
        try:
            return ASUStructureView(path).unview()
        except ValueError as error:
            if "repair=True" not in str(error):
                raise
            return ASUStructureView(load(str(path), repair=True)).unview()
    if corpus in ("icsd-standardized", "icsd-experimental"):
        try:
            return load(str(path), allow_large_cif_uncertainty=True)
        except ValueError as error:
            if "repair=True" not in str(error):
                raise
            return load(str(path), repair=True, allow_large_cif_uncertainty=True)
    raise ValueError(f"unknown corpus: {corpus}")


def _unitcell_input(source: Any) -> tuple[Any, int, int]:
    """Build the common P1 input and return it with ASU and full-cell site counts."""
    from httk.core import FracVector

    from httk.atomistic import ASUStructure, UnitcellStructureView, WyckoffSite

    if getattr(source, "molecular", False):
        raise ValueError("molecular structure is outside this benchmark's supported domain")
    if getattr(source, "assemblies", None):
        raise ValueError("assembly correlations are outside this benchmark's supported domain")
    view = UnitcellStructureView(source)
    if view.site_moments is not None:
        raise ValueError("magnetic site moments are outside this benchmark's supported domain")
    coordinates = view.sites.reduced_coords.to_fractions()
    names = view.species_at_sites
    sites = [WyckoffSite("a", FracVector(coordinate).normalize(), name) for coordinate, name in zip(coordinates, names)]
    value = ASUStructure(
        view.cell,
        1,
        sites,
        view.species,
        coordinate_precision=view.sites.precision,
        charge=view.charge,
    )
    return value, len(getattr(source, "wyckoff_sites", ())), len(sites)


def _seed_unimodular(rng: random.Random) -> Any:
    from httk.core import FracVector

    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for _ in range(4):
        i, j = rng.sample(range(3), 2)
        multiple = rng.choice((-2, -1, 1, 2))
        for column in range(3):
            matrix[i][column] += multiple * matrix[j][column]
    result = FracVector(matrix)
    if result.det() != 1:
        raise AssertionError("elementary shear product must have determinant +1")
    return result


def _scramble(reference: Any, seed: int) -> tuple[Any, tuple[Any, Any, list[int]]]:
    from httk.core import FracVector, SurdVector

    from httk.atomistic import ASUStructure, Cell, WyckoffSite

    rng = random.Random(seed)
    matrix = _seed_unimodular(rng)
    inverse = matrix.inv()
    denominator = rng.choice((2, 3, 4, 5, 6))
    shift = FracVector([Fraction(rng.randrange(denominator), denominator) for _ in range(3)])
    coordinates = [site.free_params for site in reference.wyckoff_sites]
    # The reference is P1 with one general-position site per atom.
    transformed = [(coordinate * inverse + shift).normalize() for coordinate in coordinates]
    order = list(range(len(transformed)))
    rng.shuffle(order)
    sites = [WyckoffSite("a", transformed[index], reference.wyckoff_sites[index].species) for index in order]
    inverse_row_norm = max(sum(abs(value) for value in row) for row in inverse.to_fractions())
    cell = Cell(
        SurdVector(matrix) * reference.cell.basis,
        precision=(reference.cell.precision * max(sum(abs(value) for value in row) for row in matrix.to_fractions()))
        if reference.cell.precision is not None
        else None,
        periodicity=reference.cell.periodicity,
    )
    result = ASUStructure(
        cell,
        1,
        sites,
        reference.species,
        coordinate_precision=(reference.coordinate_precision * inverse_row_norm)
        if reference.coordinate_precision is not None
        else None,
        charge=reference.charge,
    )
    return result, (matrix, shift, order)


def _signature(value: Any) -> tuple[dict[str, Any], str]:
    payload = {
        "spacegroup": value.spacegroup.it_number,
        "setting": value.spacegroup.hall_entry,
        "transform": repr(value.transform),
        "basis": repr(value.cell.basis),
        "sites": [
            (
                site.species,
                site.wyckoff,
                repr(site.free_params),
                repr(site.representative),
                repr(site.moment),
            )
            for site in value.wyckoff_sites
        ],
        "species": [repr(species) for species in value.species],
        "charge": repr(value.charge),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def _geometry_smoke(reference: Any, scrambled: Any, transform: tuple[Any, Any, list[int]]) -> dict[str, Any]:
    matrix, shift, order = transform
    # P1-site order is the recorded permutation; undo the integer shear and common translation.
    restored = [None] * len(order)
    for output_index, source_index in enumerate(order):
        restored[source_index] = ((scrambled.wyckoff_sites[output_index].free_params - shift) * matrix).normalize()
    source_coords = [site.free_params.normalize() for site in reference.wyckoff_sites]
    coordinate_match = restored == source_coords
    source_counts = Counter(site.species for site in reference.wyckoff_sites)
    result_counts = Counter(site.species for site in scrambled.wyckoff_sites)
    return {
        "volume_equal": reference.cell.volume == scrambled.cell.volume,
        "composition_equal": source_counts == result_counts and reference.species == scrambled.species,
        "charge_equal": reference.charge == scrambled.charge,
        "coordinates_inverse_exact": coordinate_match,
    }


def _canonicalizer(method: str):
    from importlib import import_module

    module = import_module(
        "httk.atomistic.symmetry.canonical" if method == "old" else "httk.atomistic.symmetry.canonical_protostructure"
    )
    if method == "old":
        function = lambda structure, tolerance: module.canonical_asu_legacy(
            structure, tolerance=tolerance, lift=False, preserve_chirality=True
        )
    else:
        function = lambda structure, tolerance: module.canonical_asu_protostructure(
            structure, tolerance=tolerance, preserve_chirality=True
        )
    return function, Path(module.__file__)


def _worker(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("corpus", choices=_CORPORA)
    parser.add_argument("method", choices=_METHODS)
    parser.add_argument("seed", type=int)
    parser.add_argument("scrambles", type=_nonnegative_int)
    parser.add_argument("tolerance", type=_positive_float)
    parser.add_argument("timeout", type=_positive_float)
    parser.add_argument("max_atoms", type=_positive_int)
    parser.add_argument("source_sha256")
    args = parser.parse_args(arguments)
    base = {
        "corpus": args.corpus,
        "source": str(args.source),
        "source_sha256": args.source_sha256,
        "method": args.method,
        "seed": args.seed,
        "scrambles": args.scrambles,
        "tolerance": args.tolerance,
        **_environment(),
    }
    try:
        raw = _load(args.source, args.corpus)
        reference, asu_sites, atom_count = _unitcell_input(raw)
        if atom_count > args.max_atoms:
            raise ValueError(f"atom_count {atom_count} exceeds --max-atoms {args.max_atoms}")
        inputs = [("baseline", reference, None)]
        for index in range(args.scrambles):
            scrambled, transform = _scramble(reference, args.seed + index)
            smoke = _geometry_smoke(reference, scrambled, transform)
            if not all(smoke.values()):
                raise AssertionError(f"scramble geometry smoke failed: {smoke}")
            inputs.append(
                (
                    f"scramble-{index + 1}",
                    scrambled,
                    {
                        "transform": {"matrix": repr(transform[0]), "shift": repr(transform[1])},
                        "smoke": smoke,
                    },
                )
            )
        base.update({"asu_sites": asu_sites, "atom_count": atom_count, "declared_spacegroup": raw.spacegroup.it_number})
        call, method_path = _canonicalizer(args.method)
        base["method_source_sha256"] = _sha256(method_path)
    except Exception as error:
        print(
            json.dumps({**base, "call": "setup", "status": "exception", "error": f"{type(error).__name__}: {error}"}),
            flush=True,
        )
        return 0

    try:
        previous_handler = signal.signal(signal.SIGALRM, _alarm_handler) if args.timeout else None
    except (AttributeError, ValueError):
        previous_handler = None
    signatures: dict[str, dict[str, Any]] = {}
    reference_digest = None
    for input_index, (call_name, structure, preparation) in enumerate(inputs):
        row = {**base, "call": call_name, "preparation": preparation}
        started = time.perf_counter()
        try:
            if args.timeout and previous_handler is None:
                raise RuntimeError("timed benchmark requires POSIX interval timers")
            if args.timeout:
                signal.setitimer(signal.ITIMER_REAL, args.timeout)
            result = call(structure, args.tolerance)
            elapsed = time.perf_counter() - started
            if args.timeout:
                signal.setitimer(signal.ITIMER_REAL, 0)
            payload, digest = _signature(result)
            row.update(status="success", elapsed_seconds=elapsed, signature_digest=digest)
            signatures[call_name] = payload
            if call_name == "baseline":
                reference_digest = digest
                row["reference_digest"] = digest
            else:
                baseline_signature = signatures.get("baseline")
                if baseline_signature is None:
                    row.update(status="unreferenced", error="baseline did not complete", reference_digest=None)
                else:
                    mismatches = [key for key in payload if baseline_signature.get(key) != payload[key]]
                    row.update(
                        status="mismatch" if mismatches else "success",
                        reference_digest=reference_digest,
                        mismatch_fields=mismatches,
                    )
        except _Deadline:
            row.update(status="timeout", elapsed_seconds=time.perf_counter() - started)
        except Exception as error:
            row.update(
                status="exception",
                elapsed_seconds=time.perf_counter() - started,
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            if args.timeout:
                signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["status"] == "timeout":
            for pending_name, _pending_structure, _pending_preparation in inputs[input_index + 1 :]:
                print(
                    json.dumps({**base, "call": pending_name, "status": "not_run_timeout"}, sort_keys=True), flush=True
                )
            break
    if args.timeout and previous_handler is not None:
        signal.signal(signal.SIGALRM, previous_handler)
    return 0


def _site_bin(count: int) -> str:
    return "1" if count == 1 else "2" if count == 2 else "3-8" if count <= 8 else "9-32" if count <= 32 else "33+"


def _pool(rows: list[dict[str, str]], samples: int, seed: int, *, exact: bool) -> list[dict[str, str]]:
    if exact:
        return rows
    rng = random.Random(seed)
    chosen = []
    for corpus in _CORPORA:
        candidates = [row for row in rows if row["corpus"] == corpus]
        rng.shuffle(candidates)
        chosen.extend(candidates[: max(32, samples * 4)])
    return chosen


def _select(valid: list[dict[str, Any]], samples: int, seed: int, *, exact: bool) -> list[dict[str, Any]]:
    if exact:
        return valid
    rng = random.Random(seed)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in valid:
        groups[(item["declared_spacegroup"], _site_bin(item["atom_count"]))].append(item)
    for group in groups.values():
        rng.shuffle(group)
    keys = list(groups)
    rng.shuffle(keys)
    selected = []
    while len(selected) < samples and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < samples:
                selected.append(groups[key].pop())
    return selected


def _emit(output: Any, row: dict[str, Any]) -> None:
    output.write(json.dumps(row, sort_keys=True) + "\n")
    output.flush()


def _run(args: argparse.Namespace) -> int:
    rows = _discover(args.corpus_root.resolve(), args.manifest)
    candidates = _pool(rows, args.samples, args.seed, exact=args.manifest is not None)
    valid: dict[str, list[dict[str, Any]]] = {corpus: [] for corpus in _CORPORA}
    failures = []
    for row in candidates:
        path = Path(row["path"])
        try:
            actual_digest = _sha256(path)
            if row.get("sha256") and row["sha256"].lower() != actual_digest:
                failures.append({**row, "status": "sha256_mismatch", "actual_sha256": actual_digest})
                continue
            raw = _load(path, row["corpus"])
            _, asu_sites, atom_count = _unitcell_input(raw)
            item = {
                **row,
                "path": str(path),
                "sha256": actual_digest,
                "declared_spacegroup": raw.spacegroup.it_number,
                "asu_sites": asu_sites,
                "atom_count": atom_count,
            }
            if atom_count > args.max_atoms:
                item["status"] = "skip_max_atoms"
                failures.append(item)
            else:
                valid[row["corpus"]].append(item)
        except Exception as error:
            failures.append({**row, "status": "load_exception", "error": f"{type(error).__name__}: {error}"})

    selected = []
    for corpus in _CORPORA:
        selected.extend(_select(valid[corpus], args.samples, args.seed, exact=args.manifest is not None))
    input_failure = any(item.get("status") == "sha256_mismatch" for item in failures)
    if args.manifest is not None:
        input_failure |= bool(failures)
    input_failure |= not selected
    metadata = {
        "record": "selection" if args.prepare_only else "benchmark",
        "seed": args.seed,
        "samples_per_corpus": args.samples,
        "scrambles": args.scrambles,
        "timeout_seconds": args.timeout,
        "tolerance": args.tolerance,
        "max_atoms": args.max_atoms,
        "methods": args.methods,
        "candidate_pool_per_corpus": max(32, args.samples * 4) if args.manifest is None else None,
        "candidate_counts": {corpus: len(valid[corpus]) for corpus in _CORPORA},
        "selected_counts": {corpus: sum(item["corpus"] == corpus for item in selected) for corpus in _CORPORA},
        "command": sys.argv,
        **_environment(),
        "prepare_only": args.prepare_only,
    }
    output_context = args.output.open("w", encoding="utf-8") if args.output else sys.stdout
    method_failure = False
    try:
        _emit(output_context, metadata)
        for item in failures:
            if not item.get("sha256"):
                item["sha256"] = _sha256(Path(item["path"])) if Path(item["path"]).is_file() else None
            _emit(output_context, {"record": "source", **item})
        selected.sort(key=lambda item: (item["corpus"], item["path"]))
        if args.prepare_only:
            for item in selected:
                _emit(output_context, {"record": "selected", **item})
            return 1 if args.assert_results and input_failure else 0
        for case_index, item in enumerate(selected):
            path = Path(item["path"])
            digest = item.get("sha256") or _sha256(path)
            ordered_methods = list(args.methods)
            if len(ordered_methods) == 2 and case_index % 2:
                ordered_methods.reverse()
            for method in ordered_methods:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--_worker",
                    str(path),
                    item["corpus"],
                    method,
                    str(args.seed),
                    str(args.scrambles),
                    str(args.tolerance),
                    str(args.timeout),
                    str(args.max_atoms),
                    digest,
                ]
                budget = max(30.0, (args.scrambles + 1) * (args.timeout + 5.0)) if args.timeout else None
                try:
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=budget, check=False)
                    parsed_count = 0
                    for line in completed.stdout.splitlines():
                        try:
                            worker_row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        parsed_count += 1
                        worker_row.update(
                            case_index=case_index,
                            candidate={key: item[key] for key in ("declared_spacegroup", "asu_sites", "atom_count")},
                        )
                        _emit(output_context, {"record": "call", **worker_row})
                        method_failure |= worker_row.get("status") != "success"
                    if not parsed_count or completed.returncode:
                        method_failure = True
                        _emit(
                            output_context,
                            {
                                "record": "call",
                                "corpus": item["corpus"],
                                "source": str(path),
                                "source_sha256": digest,
                                "method": method,
                                "status": "worker_error",
                                "error": completed.stderr[-4000:] or f"worker exited with {completed.returncode}",
                                "case_index": case_index,
                            },
                        )
                except subprocess.TimeoutExpired:
                    method_failure = True
                    _emit(
                        output_context,
                        {
                            "record": "call",
                            "corpus": item["corpus"],
                            "source": str(path),
                            "source_sha256": digest,
                            "method": method,
                            "status": "worker_timeout",
                            "case_index": case_index,
                        },
                    )
    finally:
        if output_context is not sys.stdout:
            output_context.close()
    return 1 if args.assert_results and (method_failure or input_failure) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True, help="root containing COD/ and ICSD/2026.1/")
    parser.add_argument("--manifest", type=Path, help="CSV or JSONL with corpus,path[,sha256] fields")
    parser.add_argument("--output", type=Path, help="JSONL destination (default: stdout)")
    parser.add_argument("--samples", type=_positive_int, default=24, help="maximum selected files per corpus")
    parser.add_argument("--scrambles", type=_nonnegative_int, default=5)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--timeout", type=_positive_float, default=60.0, help="seconds per canonicalization")
    parser.add_argument("--tolerance", type=_positive_float, default=1e-3)
    parser.add_argument("--max-atoms", type=_positive_int, default=256)
    parser.add_argument("--methods", type=_methods, default=_METHODS, help="comma-separated old,new")
    parser.add_argument(
        "--assert",
        dest="assert_results",
        action="store_true",
        help="exit nonzero on any method failure, timeout, or mismatch",
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="emit selected source paths and hashes without running methods"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark or one private per-method worker."""
    actual = sys.argv[1:] if argv is None else argv
    if actual and actual[0] == "--_worker":
        return _worker(actual[1:])
    args = _parser().parse_args(actual)
    if not hasattr(signal, "setitimer"):
        raise SystemExit("timed calls require POSIX interval timers")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
