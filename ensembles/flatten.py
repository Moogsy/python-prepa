from __future__ import annotations

from typing import Union, TypeVar

T = TypeVar("T")

NestedList = Union[T, list["NestedList"]]


def flatten(li: NestedList) -> list:
    for x in li:
        if isinstance(x, list):
            yield from flatten(x) 
        else:
            yield x
