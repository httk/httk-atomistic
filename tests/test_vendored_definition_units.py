"""Vendored property definitions use resolvable units and prefixed/standard dimension names."""

import json
from pathlib import Path

import pytest
from httk.core.units import default_registry

_DIR = Path(__file__).parent.parent / "src" / "httk" / "registry" / "schemas" / "atomistic"
_STANDARD_DIMS = {
    "dim_frames",
    "dim_spatial",
    "dim_lattice",
    "dim_sites",
    "dim_species",
    "dim_species_chemical_symbols",
}
_SKIP_UNITS = {"dimensionless", "inapplicable"}


def _property_files():
    out = []
    for p in sorted(_DIR.glob("*.json")):
        if json.loads(p.read_text()).get("x-optimade-definition", {}).get("kind") == "property":
            out.append(p)
    return out


def _walk(node):
    if isinstance(node, dict):
        yield node
        for key in ("items", "properties"):
            sub = node.get(key)
            if isinstance(sub, dict):
                for v in sub.values() if key == "properties" else [sub]:
                    yield from _walk(v)
            elif isinstance(sub, list):
                for v in sub:
                    yield from _walk(v)


@pytest.mark.parametrize("path", _property_files(), ids=lambda p: p.name)
def test_units_and_dimension_names(path):
    doc = json.loads(path.read_text())
    defined = {u["symbol"] for u in doc.get("x-optimade-unit-definitions", [])}
    registry = default_registry()
    for node in _walk(doc):
        unit = node.get("x-optimade-unit")
        if unit and unit not in _SKIP_UNITS:
            for d in registry.definitions(unit):
                assert d.symbol in defined, f"{path.name}: {unit!r} uses undefined symbol {d.symbol!r}"
        for name in node.get("x-optimade-dimensions", {}).get("names", []):
            assert name is None or name.startswith("_httk_dim_") or name in _STANDARD_DIMS, (
                f"{path.name}: dimension name {name!r}"
            )
