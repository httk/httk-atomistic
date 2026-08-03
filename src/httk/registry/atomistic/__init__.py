"""Register format adapters implemented by :mod:`httk.atomistic`."""

from httk.core.register import register_format_adapter

register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic._loading:_structure_from_payload",
    formats=("cif", "vasp-poscar"),
)
register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic._loading:_structure_from_optimade_payload",
    formats=("optimade-entry",),
)
