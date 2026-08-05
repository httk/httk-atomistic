"""Register format adapters implemented by :mod:`httk.atomistic`."""

from httk.core.register import register_format_adapter

register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic._loading:_structure_from_payload",
    formats=("cif", "mcif", "vasp-poscar"),
)
from httk.core.register import register_format_serializer

register_format_serializer(
    format="cif",
    serializer="httk.atomistic._writing:_cif_payload_from_structure",
)
register_format_serializer(
    format="vasp-poscar",
    serializer="httk.atomistic._writing:_poscar_payload_from_structure",
)
register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic._loading:_structure_from_optimade_payload",
    formats=("optimade-entry",),
)
