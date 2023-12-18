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

from concurrent.futures import Future
from typing import List, Union

from typing_extensions import TypeGuard, TypeVar

T = TypeVar("FutureResult")  # type: ignore


def is_list_of_futures(
    results: Union[List[Future[T]], List[List[T]]],
) -> TypeGuard[List[Future[T]]]:
    """Check if results is a list of futures. This function narrows the type of
    `results` to `List[Future[T]]` if it is a list of futures.

    Args:
        results: A list of futures.

    Returns:
        `True` if `results` is a list of futures, `False` otherwise.
    """
    return isinstance(results, list) and isinstance(results[0], Future)
