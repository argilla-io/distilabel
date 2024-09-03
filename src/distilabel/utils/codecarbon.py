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

import importlib.util
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Union

if TYPE_CHECKING:
    from codecarbon import EmissionsTracker


def is_codecarbon_available() -> bool:
    """Check if the codecarbon library is available."""
    return importlib.util.find_spec("codecarbon") is not None


@contextmanager
def track_emissions(
    *args: Any, **kwargs: Any
) -> Generator[Union["EmissionsTracker", None], None, None]:
    """
    A context manager to track emissions if codecarbon is available.
    If codecarbon is not available, this function acts as a no-op.
    """
    tracker: Union["EmissionsTracker", None] = None

    if is_codecarbon_available():
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(*args, **kwargs)
        tracker.start()  # type: ignore

    try:
        yield tracker
    finally:
        if tracker:
            tracker.stop()
