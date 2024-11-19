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

import json
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, TypeVar

_K = TypeVar("_K")


def group_dicts(*dicts: Dict[_K, Any], flatten: bool = False) -> Dict[_K, List[Any]]:
    """Combines multiple dictionaries into a single dictionary joining the values
    as a list for each key.

    Args:
        *dicts: the dictionaries to be combined.
        flatten: whether to flatten the list of values for each key.

    Returns:
        The combined dictionary.
    """
    combined_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            combined_dict[key].append(value)

    combined_dict = dict(combined_dict)
    if flatten:
        combined_dict = {
            k: list(chain.from_iterable(v)) for k, v in combined_dict.items()
        }
    return combined_dict


def flatten_dict(x: Dict[Any, Any]) -> Dict[Any, Any]:
    return {k: json.dumps(v) if isinstance(v, dict) else v for k, v in x.items()}


def merge_dicts(*dict_lists: dict) -> list[dict]:
    """
    Merge N lists of dictionaries with matching keys.
    The keys can be any strings, but they must match across all dictionaries within each position.

    Args:
        *dict_lists: Variable number of lists of dictionaries

    Returns:
        list: Merged list of dictionaries with combined values

    Raises:
        ValueError: If lists have different lengths or dictionaries have mismatched keys
    """
    if not dict_lists:
        return []

    # Verify all lists have the same length
    first_len = len(dict_lists[0])
    if not all(len(d) == first_len for d in dict_lists):
        raise ValueError("All input lists must have the same length")

    # For each position, get keys from first list's dictionary
    result = []
    for i in range(first_len):
        # Get keys from the first dictionary at this position
        keys = set(dict_lists[0][i].keys())

        # Verify all dictionaries at this position have the same keys
        for dict_list in dict_lists:
            if set(dict_list[i].keys()) != keys:
                raise ValueError(
                    f"All dictionaries at position {i} must have the same keys"
                )

        merged_dict = {key: [] for key in keys}

        # For each dictionary at position i in all lists
        for dict_list in dict_lists:
            current_dict = dict_list[i]
            for key in keys:
                # Ensure value is a list
                value = current_dict[key]
                if not isinstance(value, list):
                    value = [value]
                merged_dict[key].extend(value)

        result.append(merged_dict)

    return result
