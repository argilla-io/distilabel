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

import hashlib
from typing import TYPE_CHECKING, Any, List, Set

from pydantic import BaseModel, Field

from distilabel.utils.serialization import TYPE_INFO_KEY

if TYPE_CHECKING:
    pass

# Add here the name of the attributes that shouldn't be used to generate the signature.
# Attributes from a `BaseModel` that is an attribute from the root class must be prefixed
# with the name of the attribute followed by an underscore. For example, if the attribute
# `jobs_ids` is an attribute from the `llm` attribute of the root class it should be added
# as `llm_jobs_ids`.
_EXCLUDE_FROM_SIGNATURE_DEFAULTS = {
    TYPE_INFO_KEY,
    "disable_cuda_device_placement",
    "input_batch_size",
    "gpu_memory_utilization",
    "resources",
    "exclude_from_signature",
    "llm_jobs_ids",
    "llm_offline_batch_generation_block_until_done",
}


class SignatureMixin(BaseModel):
    """Mixin for creating a signature (for cache) of the class.

    Attributes:
        exclude_from_signature: list of attributes to exclude from the signature.
    """

    exclude_from_signature: Set[str] = Field(
        default=_EXCLUDE_FROM_SIGNATURE_DEFAULTS, exclude=True
    )

    @property
    def signature(self) -> str:
        """Makes a signature (hash) of the class, using its attributes.

        Returns:
            signature of the class.
        """

        def flatten_dump(d: Any, parent_key: str = "", sep: str = "_") -> List:
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dump(v, new_key, sep=sep))
                elif isinstance(v, list):
                    if len(v) == 0:
                        items.append((new_key, ""))
                    elif isinstance(v[0], str):
                        items.append((new_key, "-".join(v)))
                    else:
                        for i, x in enumerate(v):
                            items.extend(flatten_dump(x, f"{new_key}-{i}", sep=sep))
                elif new_key not in self.exclude_from_signature:
                    items.append((new_key, v))
            return items

        info = []
        for name, value in flatten_dump(self.dump()):
            info.append(f"{name}-{str(value)}")

        return hashlib.sha1("-".join(info).encode()).hexdigest()
