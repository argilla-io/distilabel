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

from concurrent.futures import Future, wait
from typing import List

from typing_extensions import TypeVar

T = TypeVar("T")


def when_all_complete(futures: List[Future[T]]) -> Future[T]:
    """Returns a `Future` that will be completed when all the provided `futures` are
    completed, and it will contain the results of the `futures`.

    Args:
        futures (List[Future]): the `Future`s to wait for.

    Returns:
        Future: the `Future` that will be completed when all the provided `futures` are
            completed, and it will contain the results of the `futures`.
    """
    all_done_future = Future()
    results = [None] * len(futures)

    def check_all_done(future: Future) -> None:
        # This is done to preserve the order of the results with respect to the order
        # of the futures.
        index = futures.index(future)
        results[index] = future.result()[0]

        _, not_done = wait(futures, return_when="FIRST_COMPLETED")
        if len(not_done) == 0:
            all_done_future.set_result(results)

    for future in futures:
        future.add_done_callback(check_all_done)

    return all_done_future
