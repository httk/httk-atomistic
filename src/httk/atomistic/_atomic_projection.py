"""Loss checks shared by bare-atomic-number presentation formats."""

from typing import Any

from httk.core import unwrap


def require_bare_atomic_projection(backend: Any, target: str) -> None:
    """Reject semantic state that a primitive triple or ASE cannot represent."""
    source = unwrap(backend)
    assemblies = getattr(source, "assemblies", getattr(backend, "assemblies", None))
    if assemblies is not None:
        raise TypeError(f"This structure cannot be represented as {target} because it has assemblies")
    composition = getattr(source, "chemical_composition", getattr(backend, "chemical_composition", None))
    if composition is not None:
        raise TypeError(
            f"This structure cannot be represented as {target} because it has a declared chemical composition"
        )
    charge = getattr(source, "charge", getattr(backend, "charge", None))
    if charge is not None:
        raise ValueError(f"This structure cannot be represented as {target} because it has a charge")
    species = getattr(source, "species", getattr(backend, "species", ()))
    for item in species:
        for field in ("charges", "spins", "labels"):
            if getattr(item, field, None) is not None:
                raise ValueError(f"This structure cannot be represented as {target} because species has {field}")


__all__ = ["require_bare_atomic_projection"]
