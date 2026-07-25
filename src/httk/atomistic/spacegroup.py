"""One identified space-group setting: its symbols, operations, and Wyckoff positions.

A :class:`Spacegroup` is a *setting*, not a space-group type. SG 15 has eighteen tabulated
settings, and they are genuinely different coordinate systems: Wyckoff letter ``e`` is
``0,y,1/4`` in the reference setting ``15:b1`` but ``1/4,0,z`` in ``15:c1``. Everything a
``Spacegroup`` exposes — its symmetry operations, its Wyckoff table — is expressed in its
own setting's coordinates.

:class:`~httk.atomistic.ASUStructure` holds its Wyckoff data against the IT **standard**
setting, obtained from :meth:`Spacegroup.standard`, and reaches any other setting through
a :class:`~httk.atomistic.SettingTransform`.
"""

from functools import cached_property
from typing import Any, Self

from httk.core import FracVector

from . import data
from .affine_operation import AffineOperation
from .setting_transform import SettingTransform
from .wyckoff import WyckoffPosition, wyckoff_positions

__all__ = ["Spacegroup", "wyckoff_letter_map"]

# Free-parameter values used to probe which Wyckoff position an orbit belongs to.
#
# "Generic" has to mean generic *after* the setting transform, which is a real constraint:
# under the transform for setting 166:R the innocuous-looking triple (1/7, 2/7, 3/7) maps
# to (1/7, 4/7, 4/7), whose two equal components put it on a special position and make a
# general position look as though it had been renamed. Coprime, mutually unrelated
# denominators avoid that, and the caller verifies the degrees of freedom anyway and moves
# on to the next candidate if they disagree.
_PROBE_PARAMETERS = (
    ("1/7", "1/11", "1/13"),
    ("2/17", "3/19", "5/23"),
    ("3/29", "7/31", "11/37"),
)


class Spacegroup:
    """A tabulated space-group setting, built from the vendored symmetry data."""

    _record: dict[str, Any]

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    # --- constructors ---

    @classmethod
    def standard(cls, it_number: int) -> Self:
        """The IT standard (reference) setting for a space-group number."""
        return cls(data.standard_spacegroup_setting(it_number))

    @classmethod
    def for_hall_entry(cls, hall_entry: str) -> Self:
        """The setting named by a normalized Hall symbol, e.g. ``"-c_2yc"``."""
        return cls(data.spacegroup_setting(hall_entry=hall_entry))

    @classmethod
    def for_setting(cls, setting_it_nc: str) -> Self:
        """The setting named by IT number and coordinate-system code, e.g. ``"15:c1"``."""
        return cls(data.spacegroup_setting(setting_it_nc=setting_it_nc))

    @classmethod
    def for_hm_entry(cls, hm_entry: str) -> Self:
        """The setting named by its Hermann-Mauguin entry, e.g. ``"C 1 2/c 1"``."""
        return cls(data.spacegroup_setting(hm_entry=hm_entry))

    # --- identity ---

    @property
    def record(self) -> dict[str, Any]:
        """The raw vendored record, for fields this class does not model."""
        return self._record

    @property
    def it_number(self) -> int:
        """The International Tables space-group number, 1 to 230."""
        return self._record["it_number"]

    @property
    def setting(self) -> str:
        """The setting name — IT number plus coordinate-system code, e.g. ``"15:c1"``."""
        return self._record["setting_it_nc"]

    @property
    def hall_entry(self) -> str:
        """The normalized Hall symbol, which names the setting unambiguously."""
        return self._record["hall_entry"]

    @property
    def hall_symbol(self) -> str:
        """The Hall symbol as conventionally written."""
        return self._record["hall"]

    @property
    def hermann_mauguin(self) -> str:
        """The short Hermann-Mauguin symbol for this setting."""
        return self._record["hm_short"]

    @property
    def hermann_mauguin_full(self) -> str:
        """The full Hermann-Mauguin symbol for this setting."""
        return self._record["hm_full"]

    @property
    def crystal_system(self) -> str:
        """The crystal system, e.g. ``"monoclinic"``."""
        return self._record["crystal_system"]

    @property
    def centring_type(self) -> str:
        """The lattice centring letter, e.g. ``"P"``, ``"C"``, ``"F"``."""
        return self._record["centring_type"]

    @property
    def is_standard_setting(self) -> bool:
        """Whether this is the IT standard (reference) setting for its space-group number."""
        return bool(self._record["is_reference_setting"])

    # --- symmetry ---

    @cached_property
    def symmetry_operations(self) -> tuple[AffineOperation, ...]:
        """Every symmetry operation of the group, in this setting's coordinates.

        The full set with centring translations already folded in, so its length is the
        group order and no separate centring pass is needed.
        """
        return tuple(AffineOperation.from_symop_record(entry) for entry in self._record["symops"])

    @cached_property
    def centering_translations(self) -> tuple[FracVector, ...]:
        """The lattice centring translations, including the zero translation."""
        return tuple(FracVector.create(entry) for entry in self._record["centering_translations"])

    @cached_property
    def wyckoff(self) -> tuple[WyckoffPosition, ...]:
        """The Wyckoff positions, ordered most specific first.

        Sorted by ``(free_count, multiplicity, letter)``, so identifying a coordinate by
        walking this order returns the most specific position it lies on.
        """
        return wyckoff_positions(self._record)

    @cached_property
    def _wyckoff_by_letter(self) -> dict[str, WyckoffPosition]:
        return {position.letter: position for position in self.wyckoff}

    def wyckoff_position(self, letter: str) -> WyckoffPosition:
        """The Wyckoff position with the given letter, e.g. ``"e"``."""
        try:
            return self._wyckoff_by_letter[letter]
        except KeyError:
            available = ", ".join(sorted(self._wyckoff_by_letter))
            raise KeyError(f"space group {self.setting} has no Wyckoff letter {letter!r}; has {available}") from None

    def identify_wyckoff(self, coordinate: Any) -> tuple[WyckoffPosition, FracVector] | None:
        """The most specific Wyckoff position holding an exact coordinate, and its parameters.

        Returns ``None`` when the coordinate lies on no position, which for a complete
        table means the input was not an exact rational site of this group. Matching is
        exact: an approximate coordinate must be snapped first (see
        :class:`~httk.atomistic.ASUStructure`'s recognition path), never passed here in
        the hope that it lands.
        """
        for position in self.wyckoff:
            parameters = position.parameters_of(coordinate)
            if parameters is not None:
                return position, parameters
        return None

    # --- settings ---

    @cached_property
    def transform_from_standard(self) -> SettingTransform:
        """The change of basis taking the IT standard setting's coordinates into this one.

        The identity exactly when this *is* the standard setting.
        """
        return SettingTransform.for_hall_entry(self.hall_entry)

    def standard_setting(self) -> "Spacegroup":
        """The IT standard setting for this space-group number."""
        if self.is_standard_setting:
            return self
        return Spacegroup.standard(self.it_number)

    # --- display ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Spacegroup):
            return NotImplemented
        return self.hall_entry == other.hall_entry

    def __hash__(self) -> int:
        return hash(self.hall_entry)

    def __repr__(self) -> str:
        return f"Spacegroup({self.setting!r}, {self.hermann_mauguin!r})"


def wyckoff_letter_map(standard: Spacegroup, target: Spacegroup) -> dict[str, str]:
    """How Wyckoff letters of the standard setting name themselves in another setting.

    Almost always the identity — but not always, and the exception is silent. Across all
    3210 non-reference ``(setting, letter)`` pairs in the vendored tables, exactly one
    setting permutes letters: in ``224:1`` the standard setting's ``j`` is that setting's
    ``i`` and vice versa. So a CIF that declares site ``24i`` in setting ``224:1`` does
    *not* mean standard-setting letter ``i``, and taking the letter at face value across
    a setting boundary produces the wrong structure with no error.

    Computed rather than hard-coded, so it survives a data refresh: each standard position
    is evaluated at generic parameters, mapped through the setting transform, and
    identified in the target's own Wyckoff table.
    """
    if standard.it_number != target.it_number:
        raise ValueError(
            f"cannot map Wyckoff letters between different space groups: " f"{standard.setting} and {target.setting}"
        )
    transform = target.transform_from_standard
    mapping: dict[str, str] = {}
    for position in standard.wyckoff:
        mapping[position.letter] = _corresponding_letter(position, transform, target, standard)

    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            f"the Wyckoff letter map from {standard.setting} to {target.setting} is not a bijection: {mapping}"
        )
    return mapping


def _corresponding_letter(
    position: WyckoffPosition,
    transform: SettingTransform,
    target: Spacegroup,
    standard: Spacegroup,
) -> str:
    """The target-setting letter naming the same Wyckoff position as ``position``.

    Probes with several parameter triples and keeps the first identification whose degrees
    of freedom agree, which is what rejects a probe point that landed on a special position
    by accident.
    """
    for parameters in _PROBE_PARAMETERS:
        probe = position.representative.coordinate(parameters[: position.free_count])
        identified = target.identify_wyckoff(transform.to_setting(probe).normalize())
        if identified is not None and identified[0].free_count == position.free_count:
            return identified[0].letter
    raise ValueError(
        f"Wyckoff letter {position.letter!r} of {standard.setting} could not be matched to a position "
        f"of {target.setting} with the same degrees of freedom"
    )
