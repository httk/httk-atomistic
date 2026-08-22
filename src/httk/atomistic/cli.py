"""The ``httk symmetry`` command: inspect and canonicalize a structure's symmetry.

The subcommands are thin wrappers over the public symmetry API
(:func:`~httk.atomistic.canonical_asu`, :func:`~httk.atomistic.canonicalize`,
:func:`~httk.atomistic.rerepresent`, :func:`~httk.atomistic.list_representations`,
:func:`~httk.atomistic.recognize_asu`): they load a file, run one operation, print a
human-readable report on stdout, and optionally save the resulting structure.
Inputs are loaded with ``repair=True`` where the format supports it; repair warnings
are reported on stderr through the ``httk`` logging channel before the report.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from httk.core import load, save
from httk.core.cli import CLIContext

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    UnitcellStructure,
    UnitcellStructureView,
    canonical_asu,
    list_representations,
    recognize_asu,
    rerepresent,
    structure_tolerance,
)
from httk.atomistic.symmetry.lift import canonicalize

#: Everything a handler may raise that is the operator's problem rather than a defect.
#: Anything here is reported as ``PROGRAM: message`` and exits ``1`` -- notably ImportError
#: for a missing spglib on the tolerant paths, and ValueError for an unrelated target.
_ERRORS = (OSError, ValueError, KeyError, TypeError, ImportError)

Handler = Callable[[argparse.Namespace, CLIContext, str], int]


def _load(filename: str, *, repair: bool = True) -> Any:
    """Load a structure file, applying documented input repairs where the format supports them."""

    try:
        return load(filename, repair=repair)
    except TypeError as error:
        if "unexpected keyword argument 'repair'" not in str(error):
            raise
        return load(filename)


def _num(value: object, *, exact: bool = False) -> str:
    """Render a number either exactly or as a compact float."""

    return str(value) if exact else f"{float(value):g}"  # type: ignore[arg-type]


def _fractions(values: Sequence[Fraction], *, exact: bool = False) -> str:
    """Render free-parameter values, or a marker when there are none."""

    return ", ".join(_num(value, exact=exact) for value in values) if values else "none"


def _print_cell(view: UnitcellStructureView, *, exact: bool = False) -> None:
    lengths = [_num(length, exact=exact) for length in view.cell.lengths]
    angles = [_num(angle, exact=exact) for angle in view.cell.angles]
    print(f"  cell: a={lengths[0]} b={lengths[1]} c={lengths[2]} alpha={angles[0]} beta={angles[1]} gamma={angles[2]}")


def _print_counts(view: UnitcellStructureView) -> None:
    counts = Counter(view.species_at_sites)
    breakdown = ", ".join(f"{species} {counts[species]}" for species in sorted(counts))
    print(f"  sites: {len(view.sites)} ({breakdown})")


def _print_group(asu: ASUStructure, *, label: str = "space group") -> None:
    spacegroup = asu.spacegroup
    print(
        f"  {label}: IT {spacegroup.it_number}, H-M '{spacegroup.hermann_mauguin}', "
        f"setting {spacegroup.setting}, {spacegroup.crystal_system}"
    )


def _print_wyckoff(asu: ASUStructure, *, exact: bool = False) -> None:
    print("  wyckoff occupation:")
    for site in asu.wyckoff_sites:
        params = _fractions(list(site.free_params.to_fractions()), exact=exact)
        print(f"    {site.wyckoff}  {site.species}  free params: {params}")


def _print_structure(asu: ASUStructure, *, label: str = "space group", exact: bool = False) -> None:
    """Print the full readable report for one asymmetric-unit structure."""

    _print_group(asu, label=label)
    _print_cell(UnitcellStructureView(asu), exact=exact)
    _print_counts(UnitcellStructureView(asu))
    _print_wyckoff(asu, exact=exact)


def _asu_for_info(loaded: Any, *, tolerance: float | None) -> ASUStructure:
    """Resolve any ordinary structure through the ASU view used by symmetry reporting."""
    source = loaded
    if not isinstance(loaded, ASUStructure):
        unitcell = UnitcellStructureView(loaded)
        if unitcell.site_moments is not None:
            # ``info`` reports crystallographic spatial symmetry. Magnetic moments can lower
            # that symmetry or prevent an ASU from representing an antiferromagnetic orbit,
            # so project only that decoration away while retaining the structural backend data.
            source = UnitcellStructure(
                unitcell.cell,
                unitcell.sites,
                unitcell.species,
                unitcell.species_at_sites,
                molecular=unitcell.molecular,
                assemblies=unitcell.assemblies,
                symmetry=unitcell.symmetry,
                chemical_composition=unitcell.chemical_composition,
                chemical_formula_descriptive=unitcell.chemical_formula_descriptive,
                chemical_formula_hill=unitcell.chemical_formula_hill,
                optimization_type=unitcell.optimization_type,
                immutable_id=unitcell.immutable_id,
                last_modified=unitcell.last_modified,
                charge=unitcell.charge,
            )
    return ASUStructureView(source, tolerance=tolerance).resolve()


def _require_asu(loaded: object, operation: str) -> ASUStructure:
    if not isinstance(loaded, ASUStructure):
        raise ValueError(
            f"{operation} requires an input with declared symmetry (an ASUStructure, e.g. a CIF), "
            f"but the file loaded as {type(loaded).__name__}"
        )
    return loaded


def _save(asu: ASUStructure, destination: str) -> None:
    save(asu, destination)
    print(f"  saved: {destination}")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_info(arguments: argparse.Namespace, context: CLIContext, prog: str) -> int:
    loaded = _load(arguments.file, repair=arguments.repair)
    asu = _asu_for_info(loaded, tolerance=arguments.tolerance)
    print(f"input: {type(loaded).__name__}")
    if isinstance(loaded, ASUStructure):
        _print_structure(asu, label="declared space group", exact=arguments.exact)
    else:
        print("  declared space group: none declared")
        _print_structure(asu, label="recognized space group", exact=arguments.exact)

    if arguments.recognize:
        view = UnitcellStructureView(loaded)
        tolerance = structure_tolerance(view) if arguments.tolerance is None else arguments.tolerance
        recognized = recognize_asu(view, tolerance=tolerance)
        print(f"recognized (tolerance {_num(tolerance, exact=arguments.exact)}):")
        _print_group(recognized)
        _print_wyckoff(recognized, exact=arguments.exact)
    return 0


def _handle_canonicalize(arguments: argparse.Namespace, context: CLIContext, prog: str) -> int:
    loaded = _load(arguments.file)
    if arguments.exact:
        asu = _require_asu(loaded, "canonicalize --exact")
        result = canonicalize(asu, tolerance=arguments.tolerance, preserve_chirality=arguments.preserve_chirality).asu
        print("exact canonical form:")
    else:
        result = canonical_asu(
            loaded, tolerance=arguments.tolerance, lift=arguments.lift, preserve_chirality=arguments.preserve_chirality
        )
        print(f"canonical form (lift={arguments.lift}):")
    _print_structure(result)
    destination = _destination(arguments)
    if destination is not None:
        _save(result, destination)
    return 0


def _handle_rerepresent(arguments: argparse.Namespace, context: CLIContext, prog: str) -> int:
    loaded = _load(arguments.file)
    asu = _require_asu(loaded, "rerepresent")
    result = rerepresent(asu, arguments.target, tolerance=arguments.tolerance)
    print(f"re-represented in IT {arguments.target}:")
    _print_structure(result)
    destination = _destination(arguments)
    if destination is not None:
        _save(result, destination)
    return 0


def _handle_representations(arguments: argparse.Namespace, context: CLIContext, prog: str) -> int:
    loaded = _load(arguments.file)
    asu = _require_asu(loaded, "representations")
    representations = list_representations(asu, arguments.target, tolerance=arguments.tolerance)
    print(f"representations in IT {arguments.target}: {len(representations)}")
    for index, representation in enumerate(representations):
        print(f"[{index}]")
        _print_structure(representation)
    return 0


# ---------------------------------------------------------------------------
# Assembly and dispatch
# ---------------------------------------------------------------------------


def _add_tolerance(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tolerance",
        metavar="X",
        type=float,
        default=None,
        help="Cartesian matching tolerance (default: derived from the structure's stated precision)",
    )


def _add_output(parser: argparse.ArgumentParser) -> None:
    output = parser.add_mutually_exclusive_group()
    output.add_argument("-o", "--out", metavar="OUT", help="save one resulting structure to OUT")
    output.add_argument(
        "--out-dir",
        metavar="DIR",
        help="save each result under its input basename in DIR",
    )


def _add_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", metavar="N", type=int, required=True, help="target space group IT number")


def build_parser(program: str) -> argparse.ArgumentParser:
    """Build the ``httk symmetry`` command tree.

    :param program: Program name used by the parser in help and errors.
    :return: Configured symmetry command parser.
    """

    parser = argparse.ArgumentParser(prog=program, description="Inspect and canonicalize a structure's symmetry")
    parser.set_defaults(handler=None, help_parser=parser)
    subparsers = parser.add_subparsers(metavar="COMMAND")

    info = subparsers.add_parser("info", help="report a structure's declared and recognized symmetry")
    info.add_argument("files", metavar="FILE", nargs="+", help="one or more structure files to inspect")
    info.add_argument("--recognize", action="store_true", help="also recognize the symmetry from the geometry")
    info.add_argument("--exact", action="store_true", help="print exact rational values instead of floats")
    info.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="disable documented input repairs (repairs are enabled by default)",
    )
    _add_tolerance(info)
    info.set_defaults(handler=_handle_info, help_parser=info, repair=True)

    canon = subparsers.add_parser("canonicalize", help="canonicalize a structure's symmetry")
    canon.add_argument("files", metavar="FILE", nargs="+", help="one or more structure files to canonicalize")
    _add_output(canon)
    _add_tolerance(canon)
    canon.add_argument("--lift", action="store_true", help="search upward for higher pseudosymmetry (spglib path)")
    canon.add_argument(
        "--exact",
        action="store_true",
        help="run the exact spglib-free canonicalization (requires declared symmetry)",
    )
    canon.add_argument(
        "--preserve-chirality",
        action="store_true",
        help="keep enantiomorphic groups (default: normalize a pair to its lower-numbered member)",
    )
    canon.set_defaults(handler=_handle_canonicalize, help_parser=canon)

    rerep = subparsers.add_parser("rerepresent", help="re-represent a structure in a target space group")
    rerep.add_argument("files", metavar="FILE", nargs="+", help="one or more structure files to re-represent")
    _add_target(rerep)
    _add_output(rerep)
    _add_tolerance(rerep)
    rerep.set_defaults(handler=_handle_rerepresent, help_parser=rerep)

    reps = subparsers.add_parser("representations", help="list a structure's representations in a target group")
    reps.add_argument("files", metavar="FILE", nargs="+", help="one or more structure files to enumerate")
    _add_target(reps)
    _add_tolerance(reps)
    reps.set_defaults(handler=_handle_representations, help_parser=reps)

    return parser


def command(argv: Sequence[str], context: CLIContext) -> int:
    """Handle the registered top-level ``symmetry`` command.

    :param argv: Arguments following the symmetry command name.
    :param context: Root CLI invocation context.
    :return: Command exit status.
    """

    parser = build_parser(f"{context.program} symmetry")
    try:
        arguments = parser.parse_args(list(argv))
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else 1
    handler: Handler | None = getattr(arguments, "handler", None)
    if handler is None:
        getattr(arguments, "help_parser", parser).print_help()
        return 0
    files: list[str] = arguments.files
    if getattr(arguments, "out", None) is not None and len(files) != 1:
        print(f"{parser.prog}: -o/--out requires exactly one FILE", file=sys.stderr)
        return 2
    out_dir = getattr(arguments, "out_dir", None)
    if out_dir is not None:
        basenames = [Path(filename).name for filename in files]
        if len(basenames) != len(set(basenames)):
            print(f"{parser.prog}: --out-dir requires distinct input basenames", file=sys.stderr)
            return 2
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"{parser.prog}: {error}", file=sys.stderr)
            return 2

    failed = False
    for filename in files:
        arguments.file = filename
        if len(files) > 1:
            print(f"==> {filename} <==")
        try:
            handler(arguments, context, parser.prog)
        except _ERRORS as error:
            print(f"{parser.prog}: {filename}: {error}", file=sys.stderr)
            failed = True
    return 1 if failed else 0


def _destination(arguments: argparse.Namespace) -> str | None:
    """Return the output selected for the current input file."""

    if arguments.out is not None:
        return str(arguments.out)
    if arguments.out_dir is not None:
        return str(Path(arguments.out_dir) / Path(arguments.file).name)
    return None


if __name__ == "__main__":
    raise SystemExit(command(sys.argv[1:], CLIContext("httk", Path.cwd())))
