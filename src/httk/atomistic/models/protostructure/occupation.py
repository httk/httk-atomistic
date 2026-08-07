"""An occupied Wyckoff position without coordinates."""

from dataclasses import dataclass

from httk.atomistic.models.species.species import Species


@dataclass(frozen=True)
class WyckoffOccupation:
    """One Wyckoff orbit occupied by one (possibly disordered) species."""

    wyckoff: str
    species: Species

    def __post_init__(self) -> None:
        object.__setattr__(self, "wyckoff", str(self.wyckoff))
        species = self.species if isinstance(self.species, Species) else Species.create(self.species)
        if "X" in species.chemical_symbols or "X" in (species.attached or ()):
            raise ValueError(f"WyckoffOccupation species {species.name!r} contains unknown chemical symbol 'X'")
        object.__setattr__(self, "species", species)

    def __hash__(self) -> int:
        return hash((self.wyckoff, self.species))
