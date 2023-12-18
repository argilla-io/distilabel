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

from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Union

from typing_extensions import TypeGuard, TypeVar

T = TypeVar("T")


def is_future(obj: Union[Future[T], Any]) -> TypeGuard[Future[T]]:
    """Checks if an object is a future narrowing the type.

    Args:
        obj (Future[T]): Object to check

    Returns:
        TypeGuard[Future[T]]: True if it is a future
    """
    return isinstance(obj, Future)
