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

from typing import TYPE_CHECKING, Any, Dict, List

from distilabel.utils.imports import _ARGILLA_AVAILABLE

if _ARGILLA_AVAILABLE:
    import argilla as rg

    if TYPE_CHECKING:
        from argilla.client.feedback.schemas.types import AllowedFieldTypes


def infer_fields_from_dataset_row(
    field_names: List[str], dataset_row: Dict[str, Any]
) -> List["AllowedFieldTypes"]:
    if not _ARGILLA_AVAILABLE:
        raise ImportError(
            "In order to use any of the functions defined within `argilla_utils` you must install `argilla`"
        )
    processed_items = []
    for arg_name in field_names:
        if arg_name not in dataset_row:
            continue
        if isinstance(dataset_row[arg_name], list):
            for idx in range(1, len(dataset_row[arg_name]) + 1):
                processed_items.append(
                    rg.TextField(name=f"{arg_name}-{idx}", title=f"{arg_name}-{idx}")
                )  # type: ignore
        elif isinstance(dataset_row[arg_name], str):
            processed_items.append(rg.TextField(name=arg_name, title=arg_name))  # type: ignore
    return processed_items
