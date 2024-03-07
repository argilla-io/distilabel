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

import inspect
from typing import Any

from typing_extensions import Annotated, get_args, get_origin


def is_parameter_annotated_with(parameter: inspect.Parameter, annotation: Any) -> bool:
    """Checks if a parameter type hint is `typing.Annotated` and in that case if it contains
    `annotation` as metadata.

    Args:
        parameter: the parameter to check.
        annotation: the annotation to check.

    Returns:
        `True` if the parameter type hint is `typing.Annotated` and contains `annotation`
        as metadata, `False` otherwise.
    """
    if get_origin(parameter.annotation) is not Annotated:
        return False

    for metadata in get_args(parameter.annotation):
        if metadata == annotation:
            return True

    return False
