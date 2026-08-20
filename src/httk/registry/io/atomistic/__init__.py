"""Register format adapters, readers, and writers implemented by :mod:`httk.atomistic`."""

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

from httk.core.register import register_reader, register_writer

# httk-core sends load kwargs to readers, not format adapters, so the ``.cif`` key maps
# to the atomistic reader (which carries the atomistic override to its adapter) rather
# than a neutral reader plus the ``cif`` format adapter.
register_reader(
    name="cif",
    reader="httk.atomistic.cif_structures:_read_cif_for_atomistic",
    extensions=(".cif",),
)

register_writer(
    name="cif",
    writer="httk.atomistic.io.cif.cif_writer:_write_cif_payload",
    format="cif",
    extensions=(".cif",),
)

register_writer(
    name="poscar",
    writer="httk.atomistic.integrations.vasp.io.poscar_writer:_write_poscar_payload",
    format="vasp-poscar",
    extensions=(".poscar", ".vasp"),
    filenames=("POSCAR", "CONTCAR"),
)

register_reader(
    name="mcif",
    reader="httk.atomistic.io.cif:read_mcif_asus",
    extensions=(".mcif",),
)

register_reader(
    name="poscar",
    reader="httk.atomistic.integrations.vasp.io:read_poscar",
    extensions=(".poscar", ".vasp"),
    filenames=("POSCAR", "CONTCAR"),
)

register_reader(
    name="oszicar",
    reader="httk.atomistic.integrations.vasp.io:read_oszicar",
    extensions=(".oszicar",),
    filenames=("OSZICAR",),
)

register_reader(
    name="outcar",
    reader="httk.atomistic.integrations.vasp.io:read_outcar",
    extensions=(".outcar",),
    filenames=("OUTCAR",),
)

register_reader(
    name="potcar",
    reader="httk.atomistic.integrations.vasp.io:read_potcar_summary",
    extensions=(".potcar",),
    filenames=("POTCAR", "POTCAR.summary"),
)

register_reader(
    name="xdatcar",
    reader="httk.atomistic.integrations.vasp.io:read_xdatcar",
    extensions=(".xdatcar",),
    filenames=("XDATCAR",),
)

register_reader(
    name="wavecar",
    reader="httk.atomistic.integrations.vasp.io.wavecar:read_wavecar",
    extensions=(".wavecar",),
    filenames=("WAVECAR",),
)

register_reader(
    name="trajectory-jsonl",
    reader="httk.atomistic.io.optimade_jsonl:read_trajectory_jsonl",
    extensions=(".jsonl",),
)

register_writer(
    name="wavecar",
    writer="httk.atomistic.integrations.vasp.io.wavecar:_write_wavecar_payload",
    format="vasp-wavecar",
    extensions=(".wavecar",),
    filenames=("WAVECAR",),
)

register_writer(
    name="trajectory-jsonl",
    writer="httk.atomistic.io.optimade_jsonl:_write_trajectory_jsonl_payload",
    format="httk-trajectory-jsonl",
    extensions=(".jsonl",),
)
