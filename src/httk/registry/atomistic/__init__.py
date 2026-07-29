"""Register format adapters implemented by :mod:`httk.atomistic`."""

from httk.core import register_format_adapter

register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic.vasp_structures:structure_from_payload",
    formats=("cif", "vasp-poscar"),
)
