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
    format="mcif",
    serializer="httk.atomistic._writing:_mcif_payload_from_structure",
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
    formats=("vasp-outcar", "vasp-xdatcar", "httk-trajectory-jsonl", "lammps-dump"),
)

from httk.core.register import register_reader, register_writer

register_reader(
    name="lammps-dump",
    reader="httk.atomistic.integrations.lammps.io:read_lammps_dump",
    format="lammps-dump",
    extensions=(".lammpstrj", ".lammpsdump"),
)

# httk-core sends load kwargs to readers, not format adapters, so the ``.cif`` key maps
# to the atomistic reader (which carries the atomistic override to its adapter) rather
# than a neutral reader plus the ``cif`` format adapter.
register_reader(
    name="cif",
    reader="httk.atomistic.cif_structures:_read_cif_for_atomistic",
    format="cif",
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
    format="mcif",
    extensions=(".mcif",),
)

register_writer(
    name="mcif",
    writer="httk.atomistic.io.cif.mcif_writer:_write_mcif_payload",
    format="mcif",
    extensions=(".mcif",),
)

register_reader(
    name="poscar",
    reader="httk.atomistic.integrations.vasp.io:read_poscar",
    format="vasp-poscar",
    extensions=(".poscar", ".vasp"),
    filenames=("POSCAR", "CONTCAR"),
)

register_reader(
    name="oszicar",
    reader="httk.atomistic.integrations.vasp.io:read_oszicar",
    format="vasp-oszicar",
    extensions=(".oszicar",),
    filenames=("OSZICAR",),
)

register_reader(
    name="outcar",
    reader="httk.atomistic.integrations.vasp.io:read_outcar",
    format="vasp-outcar",
    extensions=(".outcar",),
    filenames=("OUTCAR",),
)

register_reader(
    name="potcar",
    reader="httk.atomistic.integrations.vasp.io:read_potcar_summary",
    format="vasp-potcar",
    extensions=(".potcar",),
    filenames=("POTCAR", "POTCAR.summary"),
)

register_reader(
    name="xdatcar",
    reader="httk.atomistic.integrations.vasp.io:read_xdatcar",
    format="vasp-xdatcar",
    extensions=(".xdatcar",),
    filenames=("XDATCAR",),
)

register_reader(
    name="wavecar",
    reader="httk.atomistic.integrations.vasp.io.wavecar:read_wavecar",
    format="vasp-wavecar",
    extensions=(".wavecar",),
    filenames=("WAVECAR",),
)

register_reader(
    name="trajectory-jsonl",
    reader="httk.atomistic.io.optimade_jsonl:read_trajectory_jsonl",
    format="httk-trajectory-jsonl",
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

# --- lammps data structure formats ---
register_format_adapter(
    name="lammps-structures",
    adapter="httk.atomistic.integrations._geometry:_structure_from_geometry",
    formats=("lammps-data",),
)
register_format_serializer(
    format="lammps-data",
    serializer="httk.atomistic.integrations.lammps.io.data:_lammps_data_payload_from_structure",
)
register_reader(
    name="lammps-data",
    reader="httk.atomistic.integrations.lammps.io:read_lammps_data",
    format="lammps-data",
)
register_writer(
    name="lammps-data",
    writer="httk.atomistic.integrations.lammps.io.data:_write_lammps_data_payload",
    format="lammps-data",
    extensions=(".lmp",),
)

# --- cp2k structure formats ---
# ``.inp`` (ORCA) and ``.restart`` (LAMMPS) are shared, so CP2K claims no extension:
# load(path, format="cp2k-input") / save(obj, path, format="cp2k-input").
register_format_adapter(
    name="cp2k-structures",
    adapter="httk.atomistic.integrations._geometry:_structure_from_geometry",
    formats=("cp2k-input",),
)
register_format_serializer(
    format="cp2k-input",
    serializer="httk.atomistic.integrations.cp2k.io.cp2k_input:_cp2k_input_payload_from_structure",
)
register_reader(
    name="cp2k-input",
    reader="httk.atomistic.integrations.cp2k.io:read_cp2k_input",
    format="cp2k-input",
)
register_writer(
    name="cp2k-input",
    writer="httk.atomistic.integrations.cp2k.io.cp2k_input:_write_cp2k_input_payload",
    format="cp2k-input",
)

# --- qe structure formats ---
register_format_adapter(
    name="qe-structures",
    adapter="httk.atomistic.integrations._geometry:_structure_from_geometry",
    formats=("qe-pw-input",),
)
register_format_adapter(
    name="qe-trajectories",
    adapter="httk.atomistic.integrations._geometry:_trajectory_from_frames",
    formats=("qe-pw-output",),
)
register_format_serializer(
    format="qe-pw-input",
    serializer="httk.atomistic.integrations.qe.io.pw_input:_pw_input_payload_from_structure",
)
register_reader(
    name="qe-pw-input",
    reader="httk.atomistic.integrations.qe.io:read_pw_input",
    format="qe-pw-input",
    extensions=(".pwi",),
)
register_writer(
    name="qe-pw-input",
    writer="httk.atomistic.integrations.qe.io.pw_input:_write_pw_input_payload",
    format="qe-pw-input",
    extensions=(".pwi",),
)
register_reader(
    name="qe-pw-output",
    reader="httk.atomistic.integrations.qe.io:read_pw_output",
    format="qe-pw-output",
    extensions=(".pwo",),
)

# --- abinit structure formats ---
register_format_adapter(
    name="abinit-structures",
    adapter="httk.atomistic.integrations._geometry:_structure_from_geometry",
    formats=("abinit-input",),
)
register_format_adapter(
    name="abinit-trajectories",
    adapter="httk.atomistic.integrations._geometry:_trajectory_from_frames",
    formats=("abinit-output",),
)
register_format_serializer(
    format="abinit-input",
    serializer="httk.atomistic.integrations.abinit.io.inputs:_abinit_input_payload_from_structure",
)
register_reader(
    name="abinit-input",
    reader="httk.atomistic.integrations.abinit.io:read_abinit_input",
    format="abinit-input",
    extensions=(".abi",),
)
register_writer(
    name="abinit-input",
    writer="httk.atomistic.integrations.abinit.io.inputs:_write_abinit_input_payload",
    format="abinit-input",
    extensions=(".abi",),
)
register_reader(
    name="abinit-output",
    reader="httk.atomistic.integrations.abinit.io:read_abinit_output",
    format="abinit-output",
    extensions=(".abo",),
)

# The lossless exact structure format. Core dispatches on the final suffix only, so the
# format takes a distinct extension rather than the generic ".json"; any other filename
# (e.g. "x.httk.json") works through load/save(..., format="httk-structure").
register_reader(
    name="httk-structure",
    reader="httk.atomistic.io.httk_structure:read_httk_structure",
    format="httk-structure",
    extensions=(".httkstruct",),
)
register_writer(
    name="httk-structure",
    writer="httk.atomistic.io.httk_structure:_write_httk_structure",
    format="httk-structure",
    extensions=(".httkstruct",),
)
register_format_serializer(
    format="httk-structure",
    serializer="httk.atomistic.io.httk_structure:_structure_document",
)
register_format_adapter(
    name="httk-structure",
    adapter="httk.atomistic.io.httk_structure:_structure_from_document",
    formats=("httk-structure",),
)
