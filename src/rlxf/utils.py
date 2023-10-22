from collections import defaultdict
from typing import Any


def combine_dicts(*dicts: Any):
    combined_dict = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            combined_dict[key].append(value)
    return dict(combined_dict)
