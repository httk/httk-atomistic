"""An occupied Wyckoff position labelled by an anonymous species class."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrototemplateOccupation:
    """Store one Wyckoff orbit assigned to one anonymous species class.

    Unlike :class:`~httk.atomistic.models.protostructure.occupation.WyckoffOccupation`,
    the class carries an anonymous class label (``"A"``, ``"B"``, ...) rather than a real
    :class:`~httk.atomistic.Species`, so it can represent an element-agnostic template.

    :param wyckoff: The Wyckoff letter in the standard setting.
    :param label: The anonymous species-class label.
    """

    wyckoff: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "wyckoff", str(self.wyckoff))
        object.__setattr__(self, "label", str(self.label))
