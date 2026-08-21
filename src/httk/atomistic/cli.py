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
from typing import Any

from httk.core import load, save
from httk.core.cli import CLIContext

from httk.atomistic import (
    ASUStructure,
    UnitcellStructureView,
    canonical_asu,
    list_representations,
    recognize_asu,
    rerepresent,
    structure_tolerance,
)
from httk.atomistic.symmetry.lift import canonicalize

#: Everything a handler may raise that is the operator's problem rather than a defect.
#: Anything here is reported as ``PROGRAM: message`` and exits ``2`` -- notably ImportError
#: for a missing spglib on the tolerant paths, and ValueError for an unrelated target.
_ERRORS = (OSError, ValueError, KeyError, TypeError, ImportError)

Handler = Callable[[argparse.Namespace, CLIContext, str], int]


def _load(filename: str) -> Any:
    """Load a structure file, applying documented input repairs where the format supports them."""

    try:
        return load(filename, repair=True)
    except TypeError as error:
        if "unexpected keyword argument 'repair'" not in str(error):
            raise
        return load(filename)


def _num(value: object) -> str:
    """Render a cell length or angle compactly and stably (``5.0`` -> ``5``)."""

    return f"{float(value):g}"  # type: ignore[arg-type]


def _fractions(values: Sequence[Fraction]) -> str:
    """Render exact free-parameter values, or a marker when there are none."""

    return ", ".join(str(value) for value in values) if values else "none"


def _print_cell(view: UnitcellStructureView) -> None:
    lengths = [_num(length) for length in view.cell.lengths]
    angles = [_num(angle) for angle in view.cell.angles]
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


def _print_wyckoff(asu: ASUStructure) -> None:
    print("  wyckoff occupation:")
    for site in asu.wyckoff_sites:
        params = _fractions(list(site.free_params.to_fractions()))
        print(f"    {site.wyckoff}  {site.species}  free params: {params}")


def _print_structure(asu: ASUStructure, *, label: str = "space group") -> None:
    """Print the full readable report for one asymmetric-unit structure."""

    _print_group(asu, label=label)
    _print_cell(UnitcellStructureView(asu))
    _print_counts(UnitcellStructureView(asu))
    _print_wyckoff(asu)


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
    loaded = _load(arguments.file)
    view = UnitcellStructureView(loaded)
    print(f"input: {type(loaded).__name__}")
    if isinstance(loaded, ASUStructure):
        _print_group(loaded, label="declared space group")
        _print_cell(view)
        _print_counts(view)
        _print_wyckoff(loaded)
    else:
        print("  declared space group: none declared")
        _print_cell(view)
        _print_counts(view)

    if arguments.recognize:
        tolerance = structure_tolerance(view) if arguments.tolerance is None else arguments.tolerance
        recognized = recognize_asu(view, tolerance=tolerance)
        print(f"recognized (tolerance {_num(tolerance)}):")
        _print_group(recognized)
        _print_wyckoff(recognized)
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
    if arguments.out is not None:
        _save(result, arguments.out)
    return 0


def _handle_rerepresent(arguments: argparse.Namespace, context: CLIContext, prog: str) -> int:
    loaded = _load(arguments.file)
    asu = _require_asu(loaded, "rerepresent")
    result = rerepresent(asu, arguments.target, tolerance=arguments.tolerance)
    print(f"re-represented in IT {arguments.target}:")
    _print_structure(result)
    if arguments.out is not None:
        _save(result, arguments.out)
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
    parser.add_argument("-o", "--out", metavar="OUT", help="save the resulting structure to OUT (e.g. a CIF)")


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
    info.add_argument("file", metavar="FILE", help="the structure file to inspect")
    info.add_argument("--recognize", action="store_true", help="also recognize the symmetry from the geometry")
    _add_tolerance(info)
    info.set_defaults(handler=_handle_info, help_parser=info)

    canon = subparsers.add_parser("canonicalize", help="canonicalize a structure's symmetry")
    canon.add_argument("file", metavar="FILE", help="the structure file to canonicalize")
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
    rerep.add_argument("file", metavar="FILE", help="the structure file to re-represent")
    _add_target(rerep)
    _add_output(rerep)
    _add_tolerance(rerep)
    rerep.set_defaults(handler=_handle_rerepresent, help_parser=rerep)

    reps = subparsers.add_parser("representations", help="list a structure's representations in a target group")
    reps.add_argument("file", metavar="FILE", help="the structure file to enumerate")
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
    try:
        return handler(arguments, context, parser.prog)
    except _ERRORS as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    from pathlib import Path

    raise SystemExit(command(sys.argv[1:], CLIContext("httk", Path.cwd())))
