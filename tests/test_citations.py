import subprocess
import sys

import pytest


def _run_isolated(script: str) -> None:
    subprocess.run([sys.executable, "-c", script], check=True)


def test_basics_dataset_credit_is_lazy_and_registered() -> None:
    _run_isolated(
        """
from httk.core import credits
import httk.atomistic.data as data

heading = "Vendored crystallographic symmetry datasets (CC BY 4.0)"
assert heading not in credits.entries()
data.spacegroup_settings()
entries = credits.entries()
assert heading in entries
assert len(entries[heading]) == 1
"""
    )


def test_transforms_dataset_credit_is_lazy_and_registered() -> None:
    _run_isolated(
        """
from httk.core import credits
import httk.atomistic.data as data

heading = "Vendored crystallographic symmetry datasets (CC BY 4.0)"
assert heading not in credits.entries()
data.setting_transform("p_1")
assert heading in credits.entries()
"""
    )


def test_spglib_credit_is_registered_when_recognition_runs() -> None:
    pytest.importorskip("spglib")
    _run_isolated(
        """
from httk.core import FracVector, credits
from httk.atomistic import ASUStructure, Species, UnitcellStructureView, WyckoffSite, recognize_asu

heading = "Symmetry recognition uses spglib"
assert heading not in credits.entries()
no_parameters = FracVector.create(())
structure = ASUStructure(
    [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
    225,
    [WyckoffSite("a", no_parameters, "Na"), WyckoffSite("b", no_parameters, "Cl")],
    [
        Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,)),
        Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,)),
    ],
)
recognize_asu(UnitcellStructureView(structure))
assert heading in credits.entries()
"""
    )


def test_ase_credit_is_registered_when_ase_view_is_imported() -> None:
    pytest.importorskip("ase")
    _run_isolated(
        """
from httk.core import credits

heading = "Structure interchange with the Atomic Simulation Environment (ASE)"
assert heading not in credits.entries()
import httk.atomistic.compat.ase.view
assert heading in credits.entries()
"""
    )


def test_asymmetric_unit_credit_is_registered_with_atomistic_import() -> None:
    _run_isolated(
        """
from httk.core import credits

heading = "The symmetry and asymmetric-unit structure handling was informed by Edvard Valentin's subgroup-matching work for httk v1"
assert heading not in credits.entries()
import httk.atomistic
assert heading in credits.entries()
"""
    )
