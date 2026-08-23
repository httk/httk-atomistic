"""The structure-to-protostructure recognition adapter."""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.crystaltemplate.backend import CrystalTemplateBackend
from httk.atomistic.models.crystaltemplate.view_base import CrystalTemplateViewBase
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_precision,
)
from httk.atomistic.symmetry.recognition import recognize_asu
from httk.atomistic.symmetry.setting_transform import SettingTransform


class RecognizedProtostructure(ProtostructureBackend):
    r"""Project an ordinary structure lazily to a protostructure.

    :param obj: The ordinary structure to recognize.
    :param \*\*hints: Backend-selection hints.
    """

    kind = "structure"
    _structure: StructureBackend | None
    _setting: Any
    _standard: Any
    _transform: Any
    _tolerance: float | None
    _limit_denominator: int | None

    @staticmethod
    def _source_hints(hints: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in hints.items()
            if name not in {"kind", "setting", "standard", "transform", "tolerance", "limit_denominator"}
        }

    @staticmethod
    def _has_existing_asu(obj: Any) -> bool:
        if isinstance(obj, FundamentalDomainStructure):
            return True
        if not isinstance(obj, StructureView):
            return False
        from httk.atomistic.models.structure.asu_view import ASUStructureView

        return isinstance(obj, ASUStructureView) or isinstance(getattr(obj._backend, "_view", None), ASUStructureView)

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt and validate a recognized protostructure source.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "structure") != "structure":
            return None
        setting = hints.get("setting")
        standard = hints.get("standard")
        transform = hints.get("transform")
        if cls._has_existing_asu(obj) and any(
            hints.get(name) is not None
            for name in ("setting", "standard", "transform", "tolerance", "limit_denominator")
        ):
            raise ValueError("ProtostructureView recognition arguments cannot be used with an existing ASU")
        if setting is not None and (standard is not None or transform is not None):
            raise TypeError("recognize_asu() takes either 'setting' or 'standard'/'transform', not both")
        if setting is None and (standard is not None or transform is not None):
            if standard is None or transform is None:
                raise TypeError("recognize_asu() needs both 'standard' and 'transform' when either is given")
            if not standard.is_standard_setting:
                raise ValueError(f"'standard' must be an IT standard setting, got {standard.setting}")
        if isinstance(obj, (CrystalTemplateBackend, CrystalTemplateViewBase)):
            return None
        if isinstance(obj, (ChemicalFormulaBackend, ChemicalFormulaViewBase)):
            return None
        if not isinstance(obj, (StructureView, StructureBackend)):
            source_hints = cls._source_hints(hints)
            try:
                StructureBackend._select_backend(obj, **source_hints)
            except TypeError as exc:
                # Only StructureBackend.create's own no-match error means this probe declines;
                # TypeErrors raised after a structure adapter matched are real input errors.
                if str(exc) == f"Cannot represent {type(obj)} as StructureBackend":
                    return None
                raise
        return cls(obj, **hints)

    def __init__(self, obj: Any, **hints: Any) -> None:
        if isinstance(obj, StructureView):
            self._structure = obj._backend
        elif isinstance(obj, StructureBackend):
            self._structure = obj
        else:
            self._structure = StructureBackend._select_backend(obj, **self._source_hints(hints))
        self._setting = hints.get("setting")
        self._standard = hints.get("standard")
        self._transform = hints.get("transform")
        self._tolerance = hints.get("tolerance")
        self._limit_denominator = hints.get("limit_denominator")

    def _effective_structure(self) -> Any:
        resolver = getattr(self._structure, "resolve", None)
        return resolver() if resolver is not None else self._structure

    @staticmethod
    def _validate_structure(structure: Any) -> None:
        for species in structure.species:
            if "X" in species.chemical_symbols or "X" in (species.attached or ()):
                raise ValueError(
                    f"Protostructure cannot represent structure species {species.name!r} with unknown symbol 'X'"
                )
        if getattr(structure, "assemblies", None) is not None:
            raise ValueError("Protostructure cannot represent assemblies")
        if getattr(structure, "molecular", False):
            raise ValueError("Protostructure cannot represent molecular structures")
        if getattr(structure, "chemical_composition", None) is not None:
            raise ValueError("Protostructure cannot represent chemical_composition")
        if isinstance(structure, FundamentalDomainStructure):
            has_site_moments = any(site.moment is not None for site in structure.wyckoff_sites)
        else:
            has_site_moments = getattr(structure, "site_moments", None) is not None
        if has_site_moments:
            raise ValueError("Protostructure cannot represent site_moments")

    def _has_recognition_options(self) -> bool:
        return any(
            value is not None
            for value in (self._setting, self._standard, self._transform, self._tolerance, self._limit_denominator)
        )

    @cached_property
    def _derived(self) -> Protostructure:
        structure = self._effective_structure()
        asu = structure if isinstance(structure, FundamentalDomainStructure) else getattr(structure, "asu", None)
        if asu is not None and self._has_recognition_options():
            raise ValueError("ProtostructureView recognition arguments cannot be used with an existing ASU")
        self._validate_structure(structure)
        if asu is None:
            asu = recognize_asu(
                structure,
                setting=self._setting,
                standard=self._standard,
                transform=self._transform,
                tolerance=self._tolerance,
                limit_denominator=self._limit_denominator,
            )
            self._validate_structure(asu)
        # Retain only a clean standard-setting, identity-transform representative.
        # Mapping an exact ASU directly avoids expanding the conventional cell.
        if not asu.spacegroup.is_standard_setting or not asu.transform.is_identity():
            standard, standard_sites = asu._standard_wyckoff_sites()
            basis_matrix = asu.transform.matrix.T()
            asu = FundamentalDomainStructure(
                Cell(
                    asu.transform.basis_to_standard(asu.cell.basis),
                    precision=_scaled_precision(asu.cell.precision, _matrix_row_sum_factor(basis_matrix)),
                    periodicity=asu.cell.periodicity,
                ),
                standard,
                standard_sites,
                asu.species,
                transform=SettingTransform.identity(),
                coordinate_precision=_scaled_precision(
                    asu.coordinate_precision,
                    _matrix_column_sum_factor(basis_matrix.inv()),
                ),
                chemical_formula_descriptive=asu.chemical_formula_descriptive,
                chemical_formula_hill=asu.chemical_formula_hill,
                optimization_type=asu.optimization_type,
                immutable_id=asu.immutable_id,
                last_modified=asu.last_modified,
                charge=None if asu.charge is None else asu.charge * abs(asu.transform.determinant()),
            )
        standard, sites = asu._standard_wyckoff_sites()
        species_by_name = {species.name: species for species in asu.species}
        occupations = tuple((site.wyckoff, species_by_name[site.species]) for site in sites)
        return Protostructure(standard, occupations, representative=asu)

    def resolve(self) -> Protostructure:
        """Return the complete recognized protostructure."""
        return self._derived

    @property
    def spacegroup(self):
        """Return the recognized standard-setting space group."""
        return self._derived.spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the recognized occupied Wyckoff positions."""
        return self._derived.occupations

    @property
    def representative(self):
        return self._derived.representative

    @property
    def discriminator(self):
        return self._derived.discriminator

    def unwrap(self) -> Any:
        """Return the original source (an ordinary structure or a prototype)."""
        return unwrap(self._structure)
