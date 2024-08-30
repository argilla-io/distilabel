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

from typing import TYPE_CHECKING

from distilabel.constants import DISTILABEL_METADATA_KEY
from distilabel.steps.base import Step, StepInput
from distilabel.steps.columns.utils import merge_distilabel_metadata

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class CombineOutputs(Step):
    """Combine the outputs of several upstream steps.

    `CombineOutputs` is a `Step` that takes the outputs of several upstream steps and combines
    them to generate a new dictionary with all keys/columns of the upstream steps outputs.

    Input columns:
        - dynamic (based on the upstream `Step`s): All the columns of the upstream steps outputs.

    Output columns:
        - dynamic (based on the upstream `Step`s): All the columns of the upstream steps outputs.

    Categories:
        - columns

    Examples:

        Combine dictionaries of a dataset:

        ```python
        from distilabel.steps import CombineOutputs

        combine_outputs = CombineOutputs()
        combine_outputs.load()

        result = next(
            combine_outputs.process(
                [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
                [{"c": 5, "d": 6}, {"c": 7, "d": 8}],
            )
        )
        # [
        #   {"a": 1, "b": 2, "c": 5, "d": 6},
        #   {"a": 3, "b": 4, "c": 7, "d": 8},
        # ]
        ```

        Combine upstream steps outputs in a pipeline:

        ```python
        from distilabel.pipeline import Pipeline
        from distilabel.steps import CombineOutputs

        with Pipeline() as pipeline:
            step_1 = ...
            step_2 = ...
            step_3 = ...
            combine = CombineOutputs()

            [step_1, step_2, step_3] >> combine
        ```
    """

    def process(self, *inputs: StepInput) -> "StepOutput":
        combined_outputs = []
        for output_dicts in zip(*inputs):
            combined_dict = {}
            for output_dict in output_dicts:
                combined_dict.update(
                    {
                        k: v
                        for k, v in output_dict.items()
                        if k != DISTILABEL_METADATA_KEY
                    }
                )

            if any(
                DISTILABEL_METADATA_KEY in output_dict for output_dict in output_dicts
            ):
                combined_dict[DISTILABEL_METADATA_KEY] = merge_distilabel_metadata(
                    *output_dicts
                )
            combined_outputs.append(combined_dict)

        yield combined_outputs
