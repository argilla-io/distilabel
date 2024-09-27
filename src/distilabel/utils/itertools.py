# Copyright 2023-present, Argilla, Inc.
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

from itertools import zip_longest
from typing import Any, Iterable, Literal, Tuple, TypeVar

T = TypeVar("T")


# Copy pasted from https://docs.python.org/3/library/itertools.html#itertools-recipes
# Just added the type hints and use `if`s instead of `match`
def grouper(
    iterable: Iterable[T],
    n: int,
    *,
    incomplete: Literal["fill", "strict", "ignore"] = "fill",
    fillvalue: Any = None,
) -> Iterable[Tuple[T]]:
    "Collect data into non-overlapping fixed-length chunks or blocks."
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n

    if incomplete == "fill":
        return zip_longest(*args, fillvalue=fillvalue)

    if incomplete == "strict":
        return zip(*args, strict=True)

    if incomplete == "ignore":
        return zip(*args)

    raise ValueError("Expected fill, strict, or ignore")
