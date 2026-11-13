"""Build the a2db444 store fixtures: protocol-3 stores with domain bond networks.

Run from a ``git archive a2db444`` of *httk-atomistic*, extracted into a new empty directory, with
its ``src`` first on the path and the develop *httk-core* and *httk-store* installed::

    python -I -c "import sys; sys.path.insert(0, '<archive>/src'); sys.argv = ['build', ...]; \\
        exec(open('build_a2db444_store.py').read())"

Arguments: the backend (``sqlite``, ``duckdb`` or ``mongo``), the output path (a database file; for
``mongo`` a gzipped JSON dump of the database), the ``tests/fixtures`` directory, and for ``mongo``
the server URI. The script prints the saved entries and their content ids, which the upgrade test
pins.
"""

import gzip
import json
import sys
import uuid
from pathlib import Path

from httk.core import load
from httk.core.storage import content_id
from httk.store import EntryIdScheme

import httk.atomistic
from httk.atomistic import ASUStructure, FundamentalDomainStructure, UnitcellStructureView, distance_bonds
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.storage.records import ASUStructureRecord, FundamentalDomainStructureRecord, UnitcellStructureRecord
from httk.atomistic.symmetry.shapes import _site_shapes

backend, out, fixtures = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
assert "a2db444" in httk.atomistic.__file__, httk.atomistic.__file__
assert not out.exists()

diopside = distance_bonds(load(fixtures / "cod" / "1000008.cif"))  # an ASU with a network, no supplied shapes
cod_1509688 = distance_bonds(load(fixtures / "cod" / "1509688.cif"))
baseline = _site_shapes(cod_1509688)
# supplied symbols on every shaped site but the first: covers a symbol and the "" state
first = next(i for i, symbol in enumerate(baseline) if symbol is not None)
shapes = tuple(None if i == first else symbol for i, symbol in enumerate(baseline))
values: dict[str, object] = {
    "asu_network": diopside,
    "asu_network_2": cod_1509688,
    "fd_network_shapes": FundamentalDomainStructure(
        cod_1509688.cell,
        cod_1509688.spacegroup,
        cod_1509688.wyckoff_sites,
        cod_1509688.species,
        cod_1509688.transform,
        bonds=cod_1509688.bonds,
        shapes=shapes,
    ),
    "asu_plain": ASUStructure(
        diopside.cell, diopside.spacegroup, diopside.wyckoff_sites, diopside.species, diopside.transform
    ),
    "unitcell_plain": UnitcellStructureView(diopside).unview(),
}
records = {StructureEntry: (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)}
ids = EntryIdScheme("httk.test", "1")


def save_all(store: object) -> dict[str, object]:
    saved = {}
    for name, value in values.items():
        sid = store.save(value)  # type: ignore[attr-defined]
        nbonds = None if getattr(value, "bonds", None) is None else len(value.bonds)  # type: ignore[attr-defined]
        saved[name] = {"record": type(value).__name__, "id": sid, "content_id": content_id(value), "nbonds": nbonds}
    return saved


if backend in ("sqlite", "duckdb"):
    from httk.store import Backend, SqlStore

    with getattr(Backend, backend)(out) as database:
        saved = save_all(SqlStore(database, entry_ids=ids, entry_records=records))
else:
    import pymongo
    from bson import json_util
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = pymongo.MongoClient(sys.argv[4])
    name = f"httk_a2db444_{uuid.uuid4().hex}"
    try:
        saved = save_all(MongoStore(MongoDatabase(client, name), entry_ids=ids, entry_records=records))
        database = client[name]
        dump = {
            collection: {
                "documents": list(database[collection].find()),
                "indexes": list(database[collection].list_indexes()),
            }
            for collection in sorted(database.list_collection_names())
        }
        with gzip.open(out, "wt") as handle:
            handle.write(json_util.dumps(dump, json_options=json_util.CANONICAL_JSON_OPTIONS))
    finally:
        client.drop_database(name)
        client.close()
print(json.dumps(saved, indent=2))
