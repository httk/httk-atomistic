"""Readers refuse numeric text whose exact value would exceed 4300 digits, promptly.

``Fraction("1e99999999")`` would build a hundred-million-digit integer; every reader that
parses file text exactly must reject it (and an over-long mantissa) with ``ValueError``.
"""

import time
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from io import StringIO
from pathlib import Path

import pytest
from httk.core import load

from httk.atomistic._composition_values import as_fraction, bounded_fraction
from httk.atomistic.integrations.lammps.io import LammpsDumpFile
from httk.atomistic.integrations.qe.io import read_pw_input
from httk.atomistic.integrations.vasp.io.doscar import read_doscar
from httk.atomistic.io.cif._xyz_expr import _parse_linear_expr
from httk.atomistic.io.cif.cif_parser import parse_cif_float, parse_cif_fraction, parse_cif_int
from httk.atomistic.io.httk_structure import _rational, _surd
from httk.atomistic.models.structure.optimade import OptimadeStructure
from httk.atomistic.symmetry.xyz import operation_from_xyz

TOKENS = ["1e99999999", "1E-99999999", "1" * 3000 + "." + "1" * 3000]


def _doscar(token: str) -> None:
    read_doscar(
        StringIO(f"1 1 0 0\n10.0 2.0 2.0 0.0\n0.0\nCAR\nsystem\n1.0 -1.0 1 0.125 1.0\n{token} 1 2\n"),
        spin_mode="nonmagnetic",
    )


def _lammps_dump(token: str, tmp_path: Path) -> None:
    source = tmp_path / "atoms.dump"
    source.write_text(
        f"ITEM: TIME\n{token}\nITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id type xs ys zs\n1 1 0 0 0\n",
        encoding="utf-8",
    )
    next(LammpsDumpFile(source).frames())


def _pw_input(token: str) -> None:
    read_pw_input(
        f"&system\n  nat = 1, ntyp = 1\n  ibrav = 1, A = 5\n/\n"
        f"ATOMIC_SPECIES\n Si 28.0855 Si.upf\nATOMIC_POSITIONS crystal\n Si {token} 0 0\n".splitlines(keepends=True)
    )


def _cif(token: str, tmp_path: Path) -> None:
    source = tmp_path / "bad.cif"
    source.write_text(
        "data_x\n_cell_length_a 5\n_cell_length_b 5\n_cell_length_c 5\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        f"Si1 Si {token} 0 0\n",
        encoding="utf-8",
    )
    load(source)


READERS: dict[str, Callable[[str, Path], object]] = {
    "helper": lambda token, _: bounded_fraction(token),
    "cif-float": lambda token, _: parse_cif_float(token, meta=True),
    "cif-float-esd": lambda token, _: parse_cif_float(token.replace("e", "(3)e").replace("E", "(3)E"), meta=True),
    "cif-fraction": lambda token, _: parse_cif_fraction(token),
    "cif-int": lambda token, _: parse_cif_int(token),
    "cif-load": _cif,
    "cif-xyz": lambda token, _: _parse_linear_expr(f"x+{token}", use_fractions=True),
    "symop-xyz": lambda token, _: operation_from_xyz(f"{token}*x,y,z"),
    "doscar": lambda token, _: _doscar(token),
    "lammps-dump": _lammps_dump,
    "qe-input": lambda token, _: _pw_input(token),
    "httk-structure-rational": lambda token, _: _rational(token, "charge"),
    "httk-structure-surd": lambda token, _: _surd(token, "cell"),
    "composition-text": lambda token, _: as_fraction(token),
    "composition-decimal": lambda token, _: as_fraction(Decimal(token)),
    "optimade-number": lambda token, _: OptimadeStructure._number(Decimal(token), property_name="x"),
}


@pytest.mark.parametrize("token", TOKENS, ids=["huge", "tiny", "long"])
@pytest.mark.parametrize("reader", list(READERS))
def test_oversized_numeric_text_is_refused_promptly(reader: str, token: str, tmp_path: Path) -> None:
    start = time.perf_counter()
    with pytest.raises(ValueError):
        READERS[reader](token, tmp_path)
    assert time.perf_counter() - start < 2


@pytest.mark.parametrize("reader", ["cif-load", "cif-xyz", "doscar", "lammps-dump", "qe-input", "composition-text"])
def test_fixtures_accept_an_ordinary_token(reader: str, tmp_path: Path) -> None:
    READERS[reader]("0.25", tmp_path)  # so the refusals above are the bound, not a broken fixture


def test_helper_still_parses_ordinary_text_exactly() -> None:
    assert bounded_fraction("1.5e3") == 1500
    assert bounded_fraction("-2/6") == Fraction(-1, 3)
    assert bounded_fraction("1e4299") == 10**4299
