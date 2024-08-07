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
from typing import Any, Dict, List, TypeVar

_K = TypeVar("_K")


def group_dicts(*dicts: Dict[_K, Any]) -> Dict[_K, List[Any]]:
    """Combines multiple dictionaries into a single dictionary joining the values
    as a list for each key.

    Args:
        *dicts: the dictionaries to be combined.

    Returns:
        The combined dictionary.
    """
    combined_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            combined_dict[key].append(value)
    return dict(combined_dict)


def flatten_dict(x: Dict[Any, Any]) -> Dict[Any, Any]:
    return {k: json.dumps(v) if isinstance(v, dict) else v for k, v in x.items()}
