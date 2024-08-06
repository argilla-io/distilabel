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

from typing import Any, Dict, List, Optional

from distilabel.steps.base import StepInput


def group_columns(
    *inputs: StepInput,
    group_columns: List[str],
    output_group_columns: Optional[List[str]] = None,
) -> StepInput:
    """Groups multiple list of dictionaries into a single list of dictionaries on the
    specified `group_columns`. If `group_columns` are provided, then it will also rename
    `group_columns`.

    Args:
        inputs: list of dictionaries to combine.
        group_columns: list of keys to merge on.
        output_group_columns: list of keys to rename the merge keys to. Defaults to `None`.

    Returns:
        A list of dictionaries where the values of the `group_columns` are combined into a
        list and renamed to `output_group_columns`.
    """
    if output_group_columns is not None and len(output_group_columns) != len(
        group_columns
    ):
        raise ValueError(
            "The length of output_group_columns must be the same as the length of group_columns"
        )
    if output_group_columns is None:
        output_group_columns = [f"grouped_{key}" for key in group_columns]
    group_columns_dict = dict(zip(group_columns, output_group_columns))

    result = []
    # Use zip to iterate over lists based on their index
    for dicts_at_index in zip(*inputs):
        combined_dict = {}
        # Iterate over dicts at the same index
        for d in dicts_at_index:
            # Iterate over key-value pairs in each dict
            for key, value in d.items():
                # If the key is in the merge_keys, append the value to the existing list
                if key in group_columns_dict.keys():
                    combined_dict.setdefault(group_columns_dict[key], []).append(value)
                # If the key is not in the merge_keys, create a new key-value pair
                else:
                    combined_dict[key] = value
        result.append(combined_dict)
    return result


def merge_columns(
    row: Dict[str, Any], columns: List[str], new_column: str = "combined_key"
) -> Dict[str, Any]:
    """Merge columns in a dictionary into a single column on the specified `new_column`.

    Args:
        row: Dictionary corresponding to a row in a dataset.
        columns: List of keys to merge.
        new_column: Name of the new key created.

    Returns:
        Dictionary with the new merged key.
    """
    result = row.copy()  # preserve the original dictionary
    combined = []
    for key in columns:
        to_combine = result.pop(key)
        if not isinstance(to_combine, list):
            to_combine = [to_combine]
        combined += to_combine
    result[new_column] = combined
    return result
