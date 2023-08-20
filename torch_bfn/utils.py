# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions"""

from __future__ import annotations
from typing import Any, Callable, Optional, TypeVar, Tuple, Union
from functools import wraps

T = TypeVar("T")


def exists(x: Optional[Any]) -> bool:
    return x is not None


def default(val: Optional[T], d: Union[T, Callable[[], T]]) -> T:
    if val is not None:
        return val
    return d() if callable(d) else d


def cast_tuple(t: Union[Tuple[T, ...], T], length: int = 1) -> Tuple[T, ...]:
    if isinstance(t, tuple):
        return t
    return (t,) * length


def once(fn: Callable[[Any], Any]):
    called = False

    @wraps(fn)
    def inner(x: Any):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)
