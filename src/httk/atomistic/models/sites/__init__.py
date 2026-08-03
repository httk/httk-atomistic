from typing import TYPE_CHECKING

from .api import SitesAPI
from .backend import SitesBackend
from .like import SitesLike
from .numeric import NumericSites
from .numeric_view import SitesNumericView
from .plain import PlainSites
from .plain_view import PlainSitesView
from .sites import Sites
from .view import SitesView
from .view_base import SitesViewBase

__all__ = [
    "NumericSites",
    "PlainSites",
    "PlainSitesView",
    "RecordSites",
    "Sites",
    "SitesAPI",
    "SitesBackend",
    "SitesLike",
    "SitesNumericView",
    "SitesView",
    "SitesViewBase",
]

if TYPE_CHECKING:
    from .record import RecordSites


def __getattr__(name: str) -> object:
    if name == "RecordSites":
        from .record import RecordSites

        globals()[name] = RecordSites
        return RecordSites
    raise AttributeError(name)
