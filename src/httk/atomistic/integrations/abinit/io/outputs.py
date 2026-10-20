"""ABINIT main output files (``.abo``) read as geometry trajectories."""

from typing import Any

from httk.atomistic.io import _check_as_written, _check_load_precision

from .inputs import _NAME, _SUFFIXED, Exact, Units, _geometry, _source_text, _value, _variables

FORMAT = "abinit-output"

_ECHOES = (
    "-outvars: echo values of preprocessed input variables",
    "-outvars: echo values of variables after computation",
)
# The exact structure variables of an echo; ``xangst`` is a lossy derived echo and never read.
_READ = frozenset({"acell", "natom", "ntypat", "rprim", "typat", "xcart", "xred", "znucl"})
_STRUCTURE = _READ | {"angdeg", "scalecart", "xangst"}


def _echo(lines: list[str], marker: str) -> tuple[dict[str, list[str]], Units] | None:
    """The structure variables of one ``-outvars:`` echo block, or ``None`` when it is absent."""
    start = next((index for index, line in enumerate(lines) if marker in line), None)
    if start is None:
        return None
    kept: list[str] = []
    keep = False
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("="):
            break
        body = line[1:]  # column 1 holds ABINIT's echo marker ("-", "P", ...)
        first = (body.split(maxsplit=1) or [""])[0].lower()
        if _NAME.fullmatch(first):
            match = _SUFFIXED.fullmatch(first)
            base, suffix = match.groups() if match is not None else (first, "")
            if first == "ndtset" or (suffix and base in _STRUCTURE):
                raise ValueError("multi-dataset ABINIT output is not supported")
            if first == "scalecart":
                # ponytail: how the echo splits scalecart from rprim is unverified; read it when a fixture shows it.
                raise ValueError("ABINIT output echoing scalecart is not supported")
            keep = first in _READ
        if keep:
            kept.append(body)
    values, units = _variables("\n".join(kept))
    if "xred" in values:
        values.pop("xcart", None)
        units.pop("xcart", None)
    return values, units


def _exact(echo: tuple[dict[str, list[str]], Units]) -> tuple[dict[str, list[Exact]], Units]:
    """An echo's values as exact numbers, for comparing geometries."""
    values, units = echo
    return {name: [_value(token, name) for token in tokens] for name, tokens in values.items()}, units


def read_abinit_output(source: Any, *, precision: float | None = None, as_written: bool = False) -> dict[str, Any]:
    """Read the geometries an ABINIT main output echoes into a neutral frames payload.

    The ``-outvars:`` echo of the preprocessed input variables is the first frame; the echo
    after computation is a second frame only when its geometry differs. Only ``acell``,
    ``rprim``, ``xred`` (else ``xcart``), ``natom``, ``ntypat``, ``typat`` and ``znucl`` are
    read; the rounded ``xangst`` echo never is. Per-iteration relaxation geometries are not
    read.

    :param source: A file path (compressed files are decompressed), a text stream, or an
        iterable of lines.
    :param precision: Optional Cartesian precision in Å for every frame.
    :param as_written: Already exact; accepted for a uniform load interface.
    :return: ``{"format": "abinit-output", "frames": [geometry payloads]}``.
    :raises ValueError: If the text has no preprocessed-variables echo, is a multi-dataset
        output, or echoes an unsupported geometry.
    """
    _check_load_precision(precision, "read_abinit_output")
    _check_as_written(as_written, "read_abinit_output")
    text, _ = _source_text(source)
    lines = text.splitlines()
    echoes = [_echo(lines, marker) for marker in _ECHOES]
    first, after = echoes
    if first is None:
        raise ValueError(f"not an ABINIT main output: no {_ECHOES[0]!r} block")
    distinct = [first] if after is None or _exact(after) == _exact(first) else [first, after]
    frames = [_geometry(*echo) for echo in distinct]
    if precision is not None:
        for frame in frames:
            frame["precision_override"] = precision
    return {"format": FORMAT, "frames": frames}


__all__ = ["read_abinit_output"]
