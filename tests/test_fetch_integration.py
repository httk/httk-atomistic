"""End-to-end URL and local-file structure adaptation tests."""

import io
import json
import urllib.request
from email.message import Message
from fractions import Fraction
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import load_entry_type_definition
from httk.core.optimade import OptimadeResource

from httk.atomistic import OptimadeStructure, UnitcellStructureView, build_supercell

_STRUCTURES_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"


class _Response(io.BytesIO):
    headers = Message()


def _entry_and_info() -> tuple[str, str]:
    schema = load_entry_type_definition(_STRUCTURES_ID)
    properties = {
        remote_name: {"$id": schema.properties[property_name].definition_id}
        for remote_name, property_name in {
            "remote_lattice": "lattice_vectors",
            "remote_fractional": "fractional_site_positions",
            "remote_species": "species",
            "remote_site_species": "species_at_sites",
            "remote_dimensions": "dimension_types",
        }.items()
    }
    entry = {
        "data": {
            "id": "fixture-1",
            "type": "structures",
            "attributes": {
                "remote_lattice": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
                "remote_fractional": [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
                "remote_species": [
                    {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
                    {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
                ],
                "remote_site_species": ["Na", "Cl"],
                "remote_dimensions": [1, 1, 1],
            },
        }
    }
    return json.dumps(entry), json.dumps({"data": {"properties": properties}})


def _mock_http(monkeypatch: pytest.MonkeyPatch, responses: dict[str, str]) -> None:
    real_urlopen = urllib.request.urlopen

    def fake_urlopen(url: Any, *, timeout: float | None) -> _Response:
        if url in responses:
            return _Response(responses[url].encode())
        return real_urlopen(url, timeout=timeout)  # type: ignore[return-value]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def _optimade_urls() -> tuple[str, str]:
    entry_url = "https://example.test/v1/structures/fixture-1"
    return entry_url, "https://example.test/v1/info/structures"


def test_fetch_optimade_adapts_quartet_and_supercell(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url, info_url = _optimade_urls()
    entry, info = _entry_and_info()
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})

    fetched = httk.core.fetch(entry_url)

    assert isinstance(fetched, OptimadeStructure)
    assert fetched.cell.basis.to_fractions_approx() == [[Fraction(2), 0, 0], [0, Fraction(3), 0], [0, 0, Fraction(4)]]
    assert fetched.sites.reduced_coords.to_fractions() == [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]]
    assert tuple(species.name for species in fetched.species) == ("Na", "Cl")
    assert fetched.species_at_sites == ("Na", "Cl")

    supercell = build_supercell(fetched, 2).structure
    assert len(supercell.sites) == 16
    assert supercell.cell.basis.to_fractions_approx() == [[4, 0, 0], [0, 6, 0], [0, 0, 8]]


def test_fetch_optimade_raw_and_view_preserve_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_url, info_url = _optimade_urls()
    entry, info = _entry_and_info()
    _mock_http(monkeypatch, {entry_url: entry, info_url: info})

    raw = httk.core.fetch(entry_url, raw=True)
    fetched = httk.core.fetch(entry_url)

    assert raw["format"] == "optimade-entry"
    assert isinstance(raw["resource"], OptimadeResource)
    assert isinstance(fetched, OptimadeStructure)
    assert UnitcellStructureView(fetched).unwrap() is fetched.resource


def test_load_poscar_and_fetch_file_cif_end_to_end(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        """He cell
1.0
2.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 4.0
He
1
Direct
0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    loaded = httk.core.load(str(poscar))
    supercell = build_supercell(loaded, 2).structure
    assert len(supercell.sites) == 8
    assert supercell.cell.basis.to_fractions_approx() == [[4, 0, 0], [0, 6, 0], [0, 0, 8]]

    cif = tmp_path / "same.cif"
    cif.write_text(
        """data_same
_cell_length_a 2
_cell_length_b 3
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
He1 He 0 0 0 1
""",
        encoding="utf-8",
    )
    assert httk.core.fetch(cif.as_uri()) == httk.core.load(str(cif))
