"""Vendored crystallographic symmetry datasets and their lookups.

Two datasets ship with *httk-atomistic*, both read through
:class:`~httk.core.DataLoader` and both licensed CC BY 4.0 (see the adjacent
``LICENSE`` and ``README.md`` for the full attribution):

``symmetry_basics.json.gz``
    One record per space-group **setting** — 527 of them, of which 230 are flagged
    ``is_reference_setting`` (the International Tables standard setting for their IT
    number). Each record is self-contained *in its own setting*: its symmetry operations,
    its Wyckoff table, and its asymmetric-unit region are all expressed in that setting's
    coordinates. So SG 15 Wyckoff letter ``e`` reads ``0,y,1/4`` in the reference setting
    ``15:b1`` but ``1/4,0,z`` in ``15:c1``.

``spacegroup_setting_transforms.json.gz``
    The change-of-basis operation taking each setting to its IT standard setting, keyed on
    Hall entry, covering all 527 settings. See :func:`setting_transform` for the direction
    convention, which is easy to get backwards.

Nothing here is read at import time. The first lookup triggers the load: ~0.4 s and
~12 MB resident for ``symmetry_basics``, paid once per process and only if symmetry is
actually used. The transforms dataset is negligible.

Two field-choice traps worth stating once, because both fail silently:

* Use ``symops`` and ``orbit``, not ``symops_mod_centering`` and ``orbit_mod_centering``.
  The former are the full sets with centering translations folded in, so
  ``len(orbit) == multiplicity`` holds; the ``_mod_centering`` variants are the factored
  forms and mixing the two in a set comparison misreports every centred group.
* ``orbit[0]`` follows the record's ``first_orbit``, which differs from
  ``first_orbit_ita`` in 180 of the 3440 Wyckoff entries. Both describe the same orbit,
  but only the latter matches what International Tables prints.
"""

import atexit
from contextlib import ExitStack
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from httk.core import DataLoader

__all__ = [
    "point_groups",
    "setting_transform",
    "spacegroup_setting",
    "spacegroup_settings",
    "spglib_default_spacegroup_setting",
    "standard_setting_it_numbers",
    "standard_spacegroup_setting",
]

_RESOURCES = ExitStack()
atexit.register(_RESOURCES.close)


@cache
def _resource_path(name: str) -> Path:
    """A real filesystem path for a packaged data file.

    ``importlib.resources.files`` yields a plain ``Path`` for an ordinary installation,
    but only a ``Traversable`` when the package is loaded from a zip, so this goes through
    ``as_file`` rather than assuming. The extraction that implies for a zipped install is
    kept alive for the life of the process by the module-level ``ExitStack``.
    """
    return _RESOURCES.enter_context(as_file(files(__package__).joinpath(name)))


@cache
def _basics() -> DataLoader:
    return DataLoader("httk.atomistic.symmetry_basics", _resource_path("symmetry_basics.json.gz"))


@cache
def _transforms() -> DataLoader:
    return DataLoader(
        "httk.atomistic.spacegroup_setting_transforms",
        _resource_path("spacegroup_setting_transforms.json.gz"),
    )


def _lookup_index(loader: DataLoader, dataset: str, name: str) -> dict[str, int]:
    """A named lookup index from a dataset, as a plain name-to-position mapping.

    ``DataLoader.index`` is ``None`` for a document that is not in the structured JSON-LD
    form. That cannot happen for the files vendored here, but it is worth failing with a
    sentence that names the cause rather than an ``AttributeError`` on ``None`` if a data
    refresh ever changes the shape.
    """
    index = loader.index
    if index is None:
        raise RuntimeError(f"vendored dataset {dataset!r} has no lookup indices; its file shape changed")
    return getattr(index, name)


def spacegroup_settings() -> list[dict[str, Any]]:
    """Every tabulated space-group setting, one record each (527 of them)."""
    return _basics().data.spacegroups


def point_groups() -> list[dict[str, Any]]:
    """The 32 crystallographic point groups, with their symmetry operations and character tables."""
    return _basics().data.pointgroups


def spacegroup_setting(
    *,
    hall_entry: str | None = None,
    setting_it_nc: str | None = None,
    hm_entry: str | None = None,
) -> dict[str, Any]:
    """The setting record identified by exactly one of the given keys.

    ``hall_entry`` is the normalized Hall symbol (``"-c_2yc"``), ``setting_it_nc`` the
    IT number with coordinate-system code (``"15:c1"``), and ``hm_entry`` the
    Hermann-Mauguin entry name (``"C 1 2/c 1"``). A Hall entry names a setting
    unambiguously — symbol, axes and origin — which is why it is the key the transform
    dataset uses.

    Raises :class:`KeyError` if the key is unknown, and :class:`TypeError` unless exactly
    one key is given.
    """
    given = {
        "hall_entry": hall_entry,
        "setting_it_nc": setting_it_nc,
        "hm_entry": hm_entry,
    }
    supplied = {name: value for name, value in given.items() if value is not None}
    if len(supplied) != 1:
        raise TypeError(f"spacegroup_setting() takes exactly one of {', '.join(given)}; got {len(supplied)}")

    key, value = next(iter(supplied.items()))
    index = _lookup_index(_basics(), "symmetry_basics", f"index_{key}_to_spacegroups")
    try:
        position = index[value]
    except KeyError:
        raise KeyError(f"no space-group setting with {key}={value!r}") from None
    return _basics().data.spacegroups[position]


def standard_spacegroup_setting(it_number: int) -> dict[str, Any]:
    """The IT standard (reference) setting for a space-group number, ``1 <= it_number <= 230``.

    This is the setting flagged ``is_reference_setting`` and is the one
    :func:`setting_transform` transforms to. Note it is **not** always spglib's default
    setting: the two differ for the 24 space groups with two origin choices (48, 50, 59,
    68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142, 201, 203, 222,
    224, 227, 228) and agree for the other 206. Any interoperation with spglib must go
    through an explicit transform rather than assuming the two coincide.
    """
    index = _lookup_index(_basics(), "symmetry_basics", "index_it_number_to_std_spacegroups")
    try:
        position = index[str(int(it_number))]
    except KeyError:
        raise KeyError(f"no space group with IT number {it_number!r}") from None
    return _basics().data.spacegroups[position]


def spglib_default_spacegroup_setting(it_number: int) -> dict[str, Any]:
    """The setting spglib treats as its default for a space-group number.

    This differs from :func:`standard_spacegroup_setting` for the 24 space groups with two
    origin choices and coincides with it for the other 206, which is exactly why any code
    that hands coordinates to or takes them from spglib must transform explicitly rather
    than assume the two agree — the failure mode is a structure displaced by a fraction of
    a cell that still passes a symmetry check.
    """
    index = _lookup_index(_basics(), "symmetry_basics", "index_it_number_to_spglib_default_spacegroups")
    try:
        position = index[str(int(it_number))]
    except KeyError:
        raise KeyError(f"no space group with IT number {it_number!r}") from None
    return _basics().data.spacegroups[position]


def standard_setting_it_numbers() -> list[int]:
    """The IT numbers that have a tabulated standard setting: ``[1, ..., 230]``."""
    return sorted(int(key) for key in _lookup_index(_basics(), "symmetry_basics", "index_it_number_to_std_spacegroups"))


def setting_transform(hall_entry: str) -> dict[str, Any]:
    """The change-of-basis operation between a setting and its IT standard setting.

    The returned record's ``affine_transformation`` holds an exact rational ``matrix``
    ``M`` and ``vector`` ``v``. **Direction matters and the field name is misleading**:
    despite deriving from a table called ``hall_to_it_std_transform``, the pair maps
    standard-setting coordinates *into* this setting, as column vectors::

        x_own = M @ x_std + v

    Under httk's row-vector convention that is ``f_own = f_std @ M.T + v``, with the
    reverse ``f_std = (f_own - v) @ inv(M).T`` and cell basis rows transforming as
    ``B_own = inv(M).T @ B_std``. Applying it backwards yields a structurally valid but
    systematically wrong crystal.

    ``M`` is unimodular for 520 of the 527 settings. The exceptions are the seven
    rhombohedral-axes settings (IT numbers 146, 148, 155, 160, 161, 166, 167), where
    ``det M == 3`` because the standard hexagonal cell has three times the volume of the
    rhombohedral one — and correspondingly ``inv(M)`` has thirds, so nothing may assume
    the reverse transform is integral.
    """
    index = _lookup_index(
        _transforms(), "spacegroup_setting_transforms", "index_hall_entry_to_spacegroup_setting_transforms"
    )
    try:
        position = index[hall_entry]
    except KeyError:
        raise KeyError(f"no setting transform for hall_entry={hall_entry!r}") from None
    return _transforms().data.spacegroup_setting_transforms[position]
