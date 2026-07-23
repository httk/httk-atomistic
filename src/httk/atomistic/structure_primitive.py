"""
Backend wrapping an spglib-like (lattice, positions, numbers) triple.
"""

from typing import Any

from ._vector_guards import is_basis_3x3, is_coords_nx3, try_surdvector
from .cell import Cell
from .elements import symbol_of
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_primitive_triple(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        return False
    lattice, positions, numbers = obj
    if not is_basis_3x3(lattice) or not is_coords_nx3(positions):
        return False
    if not isinstance(numbers, (list, tuple)) or not all(_is_number(z) for z in numbers):
        return False
    positions_vector = try_surdvector(positions)
    nsites = positions_vector.dim[0] if positions_vector is not None and len(positions_vector.dim) == 2 else 0
    return len(numbers) == nsites


class StructurePrimitive(StructureBackend):
    """
    Backend for a crystal structure backed by an spglib-like triple.

    The native representation is a length-3 ``(lattice, positions, numbers)`` list
    or tuple, where ``lattice`` is 3x3, ``positions`` is Nx3 reduced coordinates,
    and ``numbers`` is the length-N sequence of atomic numbers. The quartet is
    derived lazily and cached: ``cell`` is a ``Cell``, ``sites`` a ``Sites``,
    ``species`` one single-element ``Species`` per distinct atomic number, and
    ``unwrap`` returns the original triple.
    """

    _raw: Any
    _lattice: Any
    _positions: Any
    _numbers: tuple[int, ...]
    _cell_cache: Cell | None
    _sites_cache: Sites | None
    _species_cache: tuple[Species, ...] | None
    _species_at_sites_cache: tuple[str, ...] | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "primitive") != "primitive":
            return None
        if not _is_primitive_triple(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        lattice, positions, numbers = obj
        self._raw = obj
        self._lattice = lattice
        self._positions = positions
        self._numbers = tuple(int(z) for z in numbers)
        self._cell_cache = None
        self._sites_cache = None
        self._species_cache = None
        self._species_at_sites_cache = None

    @property
    def cell(self) -> Cell:
        if self._cell_cache is None:
            self._cell_cache = Cell(self._lattice)
        return self._cell_cache

    @property
    def sites(self) -> Sites:
        if self._sites_cache is None:
            self._sites_cache = Sites(self._positions)
        return self._sites_cache

    @property
    def species(self) -> tuple[Species, ...]:
        if self._species_cache is None:
            distinct: list[int] = []
            seen: set[int] = set()
            for z in self._numbers:
                if z not in seen:
                    seen.add(z)
                    distinct.append(z)
            self._species_cache = tuple(
                Species(name=symbol_of(z), chemical_symbols=(symbol_of(z),), concentration=(1.0,)) for z in distinct
            )
        return self._species_cache

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        if self._species_at_sites_cache is None:
            self._species_at_sites_cache = tuple(symbol_of(z) for z in self._numbers)
        return self._species_at_sites_cache

    def unwrap(self) -> Any:
        return self._raw
