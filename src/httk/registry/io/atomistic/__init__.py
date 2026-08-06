"""Register format adapters implemented by :mod:`httk.atomistic`."""

from httk.core.register import register_format_adapter

register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic._loading:_structure_from_payload",
    formats=("cif", "mcif", "vasp-poscar"),
)
register_format_adapter(
    name="atomistic-wavefunctions",
    adapter="httk.atomistic.wavefunction:_planewaves_from_payload",
    formats=("vasp-wavecar",),
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
register_format_serializer(
    format="vasp-wavecar",
    serializer="httk.atomistic.wavefunction:_wavecar_payload_from_planewaves",
)
register_format_serializer(
    format="httk-trajectory-jsonl",
    serializer="httk.atomistic._writing:_trajectory_jsonl_payload",
)
register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic._loading:_structure_from_optimade_payload",
    formats=("optimade-entry",),
)
register_format_adapter(
    name="atomistic-trajectories",
    adapter="httk.atomistic._loading:_trajectory_from_payload",
    formats=("vasp-outcar", "vasp-xdatcar", "httk-trajectory-jsonl"),
)
