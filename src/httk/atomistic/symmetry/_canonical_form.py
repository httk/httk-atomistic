"""The canonical-form versions of httk's canonical classification labels.

A separate module so that storage records can import them without the canonicalisers' imports.
Each family's records store their family's version in ``canonical_form``:

- labels: bare prototype and bare protostructure records (``Formulatype`` has no record; its
  site-count order enters every stored label, so it belongs to this family);
- refined: prototype and protostructure records, which may carry a representative;
- nets: protonettype and protonetstructure records.

Bump policy: a label-rule change bumps all three versions; a change to the refined
representative bumps the refined version only; a change to the net rule bumps the net
version only.
"""

_LABEL_FORM_VERSION = 4
_REFINED_FORM_VERSION = 9
_NET_FORM_VERSION = 9
