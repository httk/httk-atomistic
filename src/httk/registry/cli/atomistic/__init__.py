"""Register the ``httk symmetry`` command implemented by :mod:`httk.atomistic`.

The handler is a lazy ``"module:callable"`` reference, so root help resolves nothing
and ``httk symmetry`` imports argparse and the symmetry machinery only when it runs.
"""

from httk.core import register_cli_command

register_cli_command(
    "symmetry",
    "httk.atomistic.cli:command",
    "inspect and canonicalize a structure's symmetry",
)
