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
from typing import TYPE_CHECKING, Set

from distilabel.constants import (
    ROUTING_BATCH_FUNCTION_ATTR_NAME,
    STEP_ATTR_NAME,
)
from distilabel.utils.serialization import TYPE_INFO_KEY

if TYPE_CHECKING:
    from distilabel.pipeline.routing_batch_function import RoutingBatchFunction


class SignatureMixin:
    """Mixin for the `Pipeline` to create a signature (hash) of the pipeline,

    Attributes:
        exclude_from_signature: List of attributes to exclude from the signature.
    """

    exclude_from_signature: Set[str] = {
        TYPE_INFO_KEY,
        "disable_cuda_device_placement",
        "input_batch_size",
        "gpu_memory_utilization",
        "resources",
    }

    def _create_signature(self) -> str:
        """Makes a signature (hash) of a pipeline, using the step ids and the adjacency between them.

        The main use is to find the pipeline in the cache folder.

        Returns:
            int: Signature of the pipeline.
        """
        steps_info = []
        pipeline_dump = self.dump()["pipeline"]

        for step in pipeline_dump["steps"]:
            step_info = step["name"]
            for argument, value in sorted(step[STEP_ATTR_NAME].items()):
                if (argument in self.exclude_from_signature) or (value is None):
                    continue

                if isinstance(value, dict):
                    # input_mappings/output_mappings
                    step_info += "-".join(
                        [
                            f"{str(k)}={str(v)}"
                            for k, v in value.items()
                            if k not in self.exclude_from_signature
                        ]
                    )
                elif isinstance(value, (list, tuple)):
                    # runtime_parameters_info
                    step_info += "-".join([str(v) for v in value])
                elif isinstance(value, (int, str, float, bool)):
                    if argument not in self.exclude_from_signature:
                        # batch_size/name
                        step_info += str(value)
                else:
                    raise ValueError(
                        f"Field '{argument}' in step '{step['name']}' has type {type(value)}, explicitly cast the type to 'str'."
                    )

            steps_info.append(step_info)

        connections_info = [
            f"{c['from']}-{'-'.join(c['to'])}" for c in pipeline_dump["connections"]
        ]

        routing_batch_functions_info = []
        for function in pipeline_dump["routing_batch_functions"]:
            step = function["step"]
            routing_batch_function: "RoutingBatchFunction" = self.dag.get_step(step)[
                ROUTING_BATCH_FUNCTION_ATTR_NAME
            ]
            if type_info := routing_batch_function._get_type_info():
                step += f"-{type_info}"

        return hashlib.sha1(
            ",".join(
                steps_info + connections_info + routing_batch_functions_info
            ).encode()
        ).hexdigest()
