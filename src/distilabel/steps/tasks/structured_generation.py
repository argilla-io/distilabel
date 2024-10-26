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

import warnings
from typing import Any, Dict, List, Union

from distilabel.errors import DistilabelUserError
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.typing import StructuredInput


class StructuredGeneration(Task):
    """Generate structured content for a given `instruction` using an `LLM`.

    `StructuredGeneration` is a pre-defined task that defines the `instruction` and the `structured_output`
    as the inputs, and `generation` as the output. This task is used to generate structured content based on
    the input instruction and following the schema provided within the `structured_output` column per each
    `instruction`. The `model_name` also returned as part of the output in order to enhance it.

    Attributes:
        use_system_prompt: Whether to use the system prompt in the generation. Defaults to `True`,
            which means that if the column `system_prompt` is  defined within the input batch, then
            the `system_prompt` will be used, otherwise, it will be ignored.

    Input columns:
        - instruction (`str`): The instruction to generate structured content from.
        - structured_output (`Dict[str, Any]`): The structured_output to generate structured content from. It should be a
            Python dictionary with the keys `format` and `schema`, where `format` should be one of `json` or
            `regex`, and the `schema` should be either the JSON schema or the regex pattern, respectively.

    Output columns:
        - generation (`str`): The generated text matching the provided schema, if possible.
        - model_name (`str`): The name of the model used to generate the text.

    Categories:
        - outlines
        - structured-generation

    Examples:
        Generate structured output from a JSON schema:

        ```python
        from distilabel.steps.tasks import StructuredGeneration
        from distilabel.models import InferenceEndpointsLLM

        structured_gen = StructuredGeneration(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
            ),
        )

        structured_gen.load()

        result = next(
            structured_gen.process(
                [
                    {
                        "instruction": "Create an RPG character",
                        "structured_output": {
                            "format": "json",
                            "schema": {
                                "properties": {
                                    "name": {
                                        "title": "Name",
                                        "type": "string"
                                    },
                                    "description": {
                                        "title": "Description",
                                        "type": "string"
                                    },
                                    "role": {
                                        "title": "Role",
                                        "type": "string"
                                    },
                                    "weapon": {
                                        "title": "Weapon",
                                        "type": "string"
                                    }
                                },
                                "required": [
                                    "name",
                                    "description",
                                    "role",
                                    "weapon"
                                ],
                                "title": "Character",
                                "type": "object"
                            }
                        },
                    }
                ]
            )
        )
        ```

        Generate structured output from a regex pattern (only works with LLMs that support regex, the providers using outlines):

        ```python
        from distilabel.steps.tasks import StructuredGeneration
        from distilabel.models import InferenceEndpointsLLM

        structured_gen = StructuredGeneration(
            llm=InferenceEndpointsLLM(
                model_id="meta-llama/Meta-Llama-3-70B-Instruct",
                tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
            ),
        )

        structured_gen.load()

        result = next(
            structured_gen.process(
                [
                    {
                        "instruction": "What's the weather like today in Seattle in Celsius degrees?",
                        "structured_output": {
                            "format": "regex",
                            "schema": r"(\\d{1,2})°C"
                        },

                    }
                ]
            )
        )
        ```
    """

    use_system_prompt: bool = False

    @property
    def inputs(self) -> List[str]:
        """The input for the task are the `instruction` and the `structured_output`.
        Optionally, if the `use_system_prompt` flag is set to True, then the
        `system_prompt` will be used too."""
        columns = ["instruction", "structured_output"]
        if self.use_system_prompt:
            columns = ["system_prompt"] + columns
        return columns

    def format_input(self, input: Dict[str, Any]) -> StructuredInput:
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        if not isinstance(input["instruction"], str):
            raise DistilabelUserError(
                f"Input `instruction` must be a string. Got: {input['instruction']}.",
                page="components-gallery/tasks/structuredgeneration/",
            )

        messages = [{"role": "user", "content": input["instruction"]}]
        if self.use_system_prompt:
            if "system_prompt" in input:
                messages.insert(
                    0, {"role": "system", "content": input["system_prompt"]}
                )
            else:
                warnings.warn(
                    "`use_system_prompt` is set to `True`, but no `system_prompt` in input batch, so it will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        return (messages, input.get("structured_output", None))  # type: ignore

    @property
    def outputs(self) -> List[str]:
        """The output for the task is the `generation` and the `model_name`."""
        return ["generation", "model_name"]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a dictionary with the `generation`. The `model_name`
        will be automatically included within the `process` method of `Task`. Note that even
        if the `structured_output` is defined to produce a JSON schema, this method will return the raw
        output i.e. a string without any parsing."""
        return {"generation": output}
