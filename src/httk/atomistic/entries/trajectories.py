"""Project trajectories onto the OPTIMADE trajectories entry-provider contract."""

from collections.abc import Iterable, Mapping
from typing import Any, Self

from httk.core import EntryProvider, EntryTypeDefinition, PropertyDefinition, load_entry_type_definition

from httk.atomistic.entries.definitions import load_httk_definitions
from httk.atomistic.entries.structures import _STANDARD_PROPERTY_NAMES, _STANDARD_STRUCTURE_NAMES, _structure_projection
from httk.atomistic.models.trajectory.backend import TrajectoryBackend
from httk.atomistic.models.trajectory.like import TrajectoryLike
from httk.atomistic.models.trajectory.view import TrajectoryView
from httk.atomistic.storage.records import TrajectoryRecord

__all__ = ["TRAJECTORY_FRAME_MATERIALIZATION_LIMIT", "TrajectoryEntry", "TrajectoryEntryProvider"]

TRAJECTORY_FRAME_MATERIALIZATION_LIMIT = 100
_TRAJECTORIES_DEFINITION_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/trajectories"
_HTTK_PROPERTY_KEYS = {
    "_httk_frame_stresses": "frame_stresses",
    "_httk_frame_temperatures": "frame_temperatures",
    "_httk_frame_total_energies": "frame_total_energies",
    "_httk_time_step": "time_step",
}
_OBSERVABLE_PROPERTIES = {
    "_httk_frame_stresses": "_httk_frame_stresses",
    "_httk_frame_temperatures": "_httk_frame_temperatures",
    "_httk_frame_total_energies": "_httk_frame_total_energies",
}
_STANDARD_PROPERTY_SET = frozenset((*_STANDARD_PROPERTY_NAMES, "nframes", "reference_frames"))


def trajectory_definitions() -> dict[str, PropertyDefinition]:
    """Return the vendored httk trajectory properties keyed by served name."""
    return load_httk_definitions(_HTTK_PROPERTY_KEYS)


def _trajectories_definition() -> EntryTypeDefinition:
    return load_entry_type_definition(_TRAJECTORIES_DEFINITION_ID)


class TrajectoryEntry:
    """Non-instantiable logical family for OPTIMADE trajectory entries."""

    type = "trajectories"
    definition_id = _TRAJECTORIES_DEFINITION_ID

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("TrajectoryEntry is a logical entry family; store a trajectory representation directly")

    @classmethod
    def entry_type_definition(cls) -> EntryTypeDefinition:
        return _trajectories_definition().extended(trajectory_definitions())


def _as_trajectory(obj: TrajectoryLike | TrajectoryRecord) -> TrajectoryBackend | TrajectoryView:
    from httk.atomistic.models.trajectory.record import RecordTrajectory

    if isinstance(obj, TrajectoryRecord):
        return RecordTrajectory(obj)
    if isinstance(obj, TrajectoryView | TrajectoryBackend):
        return obj
    if isinstance(obj, Mapping):
        return TrajectoryBackend.create(obj)
    raise TypeError(f"cannot represent {type(obj).__name__} as a trajectory")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    try:
        return float(value) if value.__class__.__module__ == "fractions" else value
    except (AttributeError, TypeError, ValueError):
        return value


def _compact_or_full(name: str, values: tuple[Any, ...], definition: PropertyDefinition) -> Any:
    if not values:
        return None
    if all(value is None for value in values):
        return None
    dimensions = definition.dimensions or {}
    compactable = dimensions.get("compactable", ())
    if compactable and compactable[0] == "constant" and all(value == values[0] for value in values[1:]):
        return [_json_value(values[0])]
    return [_json_value(value) for value in values]


def _metadata(trajectory: Any) -> dict[str, Any]:
    from httk.atomistic.models.trajectory.record import RecordTrajectory

    source = trajectory.unwrap() if isinstance(trajectory, TrajectoryView) else trajectory
    if isinstance(source, RecordTrajectory):
        source = source.unwrap()
    if isinstance(source, TrajectoryRecord):
        trajectory = source
    last_modified = getattr(trajectory, "last_modified", None)
    return {
        "immutable_id": getattr(trajectory, "immutable_id", None),
        "last_modified": None if last_modified is None else last_modified.isoformat(),
    }


def _record_projection(
    trajectory: TrajectoryBackend | TrajectoryView,
    definition: EntryTypeDefinition,
) -> dict[str, Any]:
    nframes = trajectory.nframes
    values: dict[str, Any] = {
        "nframes": nframes,
        "reference_frames": (None if trajectory.reference_frames is None else list(trajectory.reference_frames)),
    }
    metadata = _metadata(trajectory)
    values.update(metadata)

    # A record is intentionally a summary backend: no standard frame property is
    # reconstructed from its bounded reference frames.
    from httk.atomistic.models.trajectory.record import RecordTrajectory

    record_backed = isinstance(trajectory, RecordTrajectory) or (
        isinstance(trajectory, TrajectoryView) and isinstance(trajectory.unwrap(), TrajectoryRecord)
    )
    if record_backed or nframes > TRAJECTORY_FRAME_MATERIALIZATION_LIMIT:
        values.update({name: None for name in _STANDARD_STRUCTURE_NAMES})
        values.update({name: None for name in _OBSERVABLE_PROPERTIES.values()})
        return values

    frames = tuple(trajectory.frames())
    projections = tuple(_structure_projection(frame) for frame in frames)
    for name in _STANDARD_STRUCTURE_NAMES:
        values[name] = _compact_or_full(name, tuple(item[name] for item in projections), definition.properties[name])

    for observable_name, property_name in _OBSERVABLE_PROPERTIES.items():
        if observable_name in trajectory.observable_names:
            values[property_name] = _compact_or_full(
                property_name,
                trajectory.observable(observable_name),
                definition.properties[property_name],
            )
        else:
            values[property_name] = None
    return values


class TrajectoryEntryProvider(EntryProvider):
    """Serve trajectory metadata and bounded frame projections.

    Native frame lists are materialized only through
    :data:`httk.atomistic.entries.trajectories.TRAJECTORY_FRAME_MATERIALIZATION_LIMIT`
    (100) frames. Larger or
    record-backed trajectories still serve the entry, frame count, references,
    and null frame properties; the full JSONL/OPTIMADE partial-data path remains
    the recovery mechanism for those frames.
    """

    def __init__(
        self,
        entries: Mapping[str, TrajectoryLike | None] | Iterable[TrajectoryLike],
        *,
        extra_definitions: Mapping[str, PropertyDefinition] | None = None,
        properties: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        normalized: dict[str, TrajectoryLike | None] = {}
        if isinstance(entries, Mapping):
            for raw_id, value in entries.items():
                entry_id = str(raw_id)
                if not entry_id:
                    raise ValueError("TrajectoryEntryProvider ids must be non-empty strings")
                normalized[entry_id] = value
        else:
            for trajectory in entries:
                candidate_id = getattr(trajectory, "id", None)
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise TypeError("iterable TrajectoryEntryProvider input must contain trajectories with an id")
                if candidate_id in normalized:
                    raise ValueError(f"duplicate trajectory id: {candidate_id!r}")
                normalized[candidate_id] = trajectory
        self._entries = normalized
        self._extra_definitions = dict(extra_definitions or {})
        definition_clashes = sorted(
            _STANDARD_PROPERTY_SET.union(_HTTK_PROPERTY_KEYS).intersection(self._extra_definitions)
        )
        if definition_clashes:
            raise ValueError(
                "custom definitions may not override standard OPTIMADE trajectory properties: "
                + ", ".join(definition_clashes)
            )
        self._properties = {str(entry_id): dict(values) for entry_id, values in (properties or {}).items()}
        used_names = sorted({name for values in self._properties.values() for name in values})
        value_clashes = sorted((_STANDARD_PROPERTY_SET | frozenset(_OBSERVABLE_PROPERTIES)).intersection(used_names))
        if value_clashes:
            raise ValueError(
                "custom values may not override standard OPTIMADE trajectory properties: " + ", ".join(value_clashes)
            )
        described = self._definition().properties
        offenders = [name for name in used_names if name not in described]
        if offenders:
            raise ValueError(
                "TrajectoryEntryProvider was given properties not described by its (extended) definition: "
                + ", ".join(offenders)
                + ". Add them via extra_definitions (custom names need a registered prefix)."
            )
        self._property_names = used_names

    def _definition(self) -> EntryTypeDefinition:
        definition = TrajectoryEntry.entry_type_definition()
        if self._extra_definitions:
            definition = definition.extended(self._extra_definitions)
        return definition

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"trajectories": self._definition()}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "trajectories":
            raise KeyError("TrajectoryEntryProvider serves only the 'trajectories' entry type.")
        property_keys = {
            name: ("__id" if name == "id" else name)
            for name in (*_STANDARD_PROPERTY_NAMES, "nframes", "reference_frames")
        }
        property_keys.update({name: name for name in _HTTK_PROPERTY_KEYS})
        property_keys.update({name: name for name in self._property_names})
        return property_keys

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "trajectories":
            raise KeyError("TrajectoryEntryProvider serves only the 'trajectories' entry type.")
        definition = self._definition()
        records: list[dict[str, Any]] = []
        for entry_id, value in self._entries.items():
            trajectory = None if value is None else _as_trajectory(value)
            record: dict[str, Any] = {
                "__id": entry_id,
                "type": TrajectoryEntry.type,
                "immutable_id": None,
                "last_modified": None,
            }
            record.update(
                {
                    "nframes": None,
                    "reference_frames": None,
                    **{name: None for name in _STANDARD_STRUCTURE_NAMES},
                    **{name: None for name in _HTTK_PROPERTY_KEYS},
                }
            )
            if trajectory is not None:
                record.update(_record_projection(trajectory, definition))
            entry_properties = self._properties.get(entry_id, {})
            for name in self._property_names:
                record[name] = entry_properties.get(name)
            records.append(record)
        return records
