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

from typing import List, Optional

from distilabel.steps.base import StepInput


def combine_dicts(
    *inputs: StepInput,
    merge_keys: List[str],
    output_merge_keys: Optional[List[str]] = None,
) -> StepInput:
    if output_merge_keys is not None and len(output_merge_keys) != len(merge_keys):
        raise ValueError(
            "The length of output_merge_keys must be the same as the length of merge_keys"
        )
    if output_merge_keys is None:
        output_merge_keys = [f"merged_{key}" for key in merge_keys]
    merge_keys_dict = dict(zip(merge_keys, output_merge_keys))

    result = []
    # Use zip to iterate over lists based on their index
    for dicts_at_index in zip(*inputs):
        combined_dict = {}
        # Iterate over dicts at the same index
        for d in dicts_at_index:
            # Iterate over key-value pairs in each dict
            for key, value in d.items():
                # If the key is in the merge_keys, append the value to the existing list
                if key in merge_keys_dict.keys():
                    combined_dict.setdefault(merge_keys_dict[key], []).append(value)
                # If the key is not in the merge_keys, create a new key-value pair
                else:
                    combined_dict[key] = value
        result.append(combined_dict)
    return result
