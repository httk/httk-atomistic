"""An occupied Wyckoff position without coordinates."""

from dataclasses import dataclass

from httk.atomistic.models.species.species import Species


@dataclass(frozen=True)
class WyckoffOccupation:
    """Store one Wyckoff orbit occupied by one possibly disordered species.

    :param wyckoff: The Wyckoff letter in the standard setting.
    :param species: The real species occupying the orbit.
    """

    wyckoff: str
    species: Species

    def __post_init__(self) -> None:
        object.__setattr__(self, "wyckoff", str(self.wyckoff))
        species = self.species if isinstance(self.species, Species) else Species.from_object(self.species)
        if "X" in species.chemical_symbols or "X" in (species.attached or ()):
            raise ValueError(f"WyckoffOccupation species {species.name!r} contains unknown chemical symbol 'X'")
        object.__setattr__(self, "species", species)

    def __hash__(self) -> int:
        return hash((self.wyckoff, self.species))
