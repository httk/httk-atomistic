"""httk-atomistic overrides httk-io's neutral ``.cif`` reader when both are installed.

httk-io registers a ``.cif`` reader that returns the neutral token tree. httk-atomistic
deliberately re-registers the same key (see :mod:`httk.registry.io.atomistic`) so that
``httk.core.load("x.cif")`` yields a native ``ASUStructure`` instead. The override only
happens when httk-io's registry package is importable, so this test is gated on that:
in httk-atomistic's own CI (no httk-io) it is skipped, and httk-io's suite skips the
mirror-image assertions when httk-atomistic is present.
"""

import importlib.util

import pytest

if importlib.util.find_spec("httk.registry.io.io") is None:
    pytest.skip("httk-io is not installed; the .cif override does not apply", allow_module_level=True)

import httk.core
from httk.core._plugins import resolve_callable


def test_cif_reader_is_overridden_to_the_atomistic_reader():
    """The registered ``.cif`` handler is httk-atomistic's, not httk-io's neutral one."""
    spec = httk.core.register.readers.require(".cif")
    assert spec.name == "atomistic-cif"
    assert spec.handler == "httk.atomistic.cif_structures:_read_cif_for_atomistic"

    fn = resolve_callable(spec.handler)
    from httk.atomistic.cif_structures import _read_cif_for_atomistic

    assert fn is _read_cif_for_atomistic
