from typing import TYPE_CHECKING

from .api import CellAPI
from .backend import CellBackend
from .cell import Cell
from .like import CellLike
from .numeric import NumericCell
from .numeric_view import CellNumericView
from .params import CellParams
from .params_view import CellParamsView
from .plain import PlainCell
from .plain_view import PlainCellView
from .view import CellView
from .view_base import CellViewBase

__all__ = [
    "Cell",
    "CellAPI",
    "CellBackend",
    "CellLike",
    "CellNumericView",
    "CellParams",
    "CellParamsView",
    "CellView",
    "CellViewBase",
    "NumericCell",
    "PlainCell",
    "PlainCellView",
    "RecordCell",
]

if TYPE_CHECKING:
    from .record import RecordCell


def __getattr__(name: str) -> object:
    if name == "RecordCell":
        from .record import RecordCell

        globals()[name] = RecordCell
        return RecordCell
    raise AttributeError(name)
