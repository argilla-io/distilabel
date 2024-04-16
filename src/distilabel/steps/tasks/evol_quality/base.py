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

from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
from pydantic import Field
from typing_extensions import override

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.evol_quality.utils import MUTATION_TEMPLATES
from distilabel.steps.tasks.typing import ChatType

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class EvolQuality(Task):
    """The `EvolQuality` task is used to evolve the quality of the responses given a prompt,
    by generating a new response with a language model. This step implements the evolution
    quality task from the paper 'What Makes Good Data for Alignment? A Comprehensive Study of
    Automatic Data Selection in Instruction Tuning'.

    Attributes:
        num_evolutions: The number of evolutions to be performed on the responses.
        store_evolutions: Whether to store all the evolved responses or just the last one.
            Defaults to `False`.
        include_original_response: Whether to include the original response within the evolved
            responses. Defaults to `False`.
        mutation_templates: The mutation templates to be used to evolve the responses.
        seed: The seed to be set for `numpy` in order to randomly pick a mutation method.
            Defaults to `42`.

    Runtime parameters:
        - `seed`: The seed to be set for `numpy` in order to randomly pick a mutation method.

    Input columns:
        - instruction (`str`): The instruction that was used to generate the `responses`.
        - response (`str`): The responses to be rewritten.

    Output columns:
        - evolved_response (`str`): The evolved response if `store_evolutions=False`.
        - evolved_responses (`List[str]`): The evolved responses if `store_evolutions=True`.
        - model_name (`str`): The name of the LLM used to evolve the responses.

    References:
        - [`What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning`](https://arxiv.org/abs/2312.15685)
    """

    num_evolutions: int
    store_evolutions: bool = False
    include_original_response: bool = False
    mutation_templates: Dict[str, str] = MUTATION_TEMPLATES

    seed: RuntimeParameter[int] = Field(
        default=42,
        description="As `numpy` is being used in order to randomly pick a mutation method, then is nice to set a random seed.",
    )

    @override
    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

    @property
    def inputs(self) -> List[str]:
        """The input for the task are the `instruction` and `response`."""
        return ["instruction", "response"]

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation. And the
        `system_prompt` is added as the first message if it exists."""
        return [{"role": "user", "content": input}]

    @property
    def outputs(self) -> List[str]:
        """The output for the task are the `evolved_response/s` and the `model_name`."""
        # TODO: having to define a `model_name` column every time as the `Task.outputs` is not ideal,
        # this could be handled always and the value could be included within the DAG validation when
        # a `Task` is used, since all the `Task` subclasses will have an `llm` with a `model_name` attr.
        _outputs = [
            ("evolved_response" if not self.store_evolutions else "evolved_responses"),
            "model_name",
        ]

        return _outputs

    def format_output(self, responses: Union[str, List[str]]) -> Dict[str, Any]:  # type: ignore
        """The output for the task is a dict with: `evolved_response` or `evolved_responses`,
        depending whether the value is either `False` or `True` for `store_evolutions`, respectively;
        and, finally, the `model_name`.

        Args:
            responses: The responses to be included within the output.

        Returns:
            if `store_evolutions=False` return {"evolved_response": ..., "model_name": ...};
            if `store_evolutions=True` return {"evolved_responses": ..., "model_name": ...}.
        """
        _output = {}

        if not self.store_evolutions:
            _output["evolved_response"] = responses[-1]
        else:
            _output["evolved_responses"] = responses

        _output["model_name"] = self.llm.model_name
        return _output

    @property
    def mutation_templates_names(self) -> List[str]:
        """Returns the names i.e. keys of the provided `mutation_templates` enum."""
        return list(self.mutation_templates.keys())

    def _apply_random_mutation(self, instruction: str, response: str) -> str:
        """Applies a random mutation from the ones provided as part of the `mutation_templates`
        enum, and returns the provided instruction within the mutation prompt.

        Args:
            instruction: The instruction to be included within the mutation prompt.

        Returns:
            A random mutation prompt with the provided instruction.
        """
        mutation = np.random.choice(self.mutation_templates_names)
        return (
            self.mutation_templates[mutation]
            .replace("<PROMPT>", instruction)
            .replace("<RESPONSE>", response[-1])
        )

    def _evolve_reponses(self, inputs: "StepInput") -> List[List[str]]:
        """Evolves the instructions provided as part of the inputs of the task.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list where each item is a list with either the last evolved instruction if
            `store_evolutions=False` or all the evolved instructions if `store_evolutions=True`.
        """
        np.random.seed(self.seed)
        instructions: List[List[str]] = [[input["instruction"]] for input in inputs]
        responses: List[List[str]] = [[input["response"]] for input in inputs]

        for iter_no in range(self.num_evolutions):
            formatted_prompts = []
            for instruction, response in zip(instructions, responses):
                formatted_prompts.append(
                    self._apply_random_mutation(instruction[-1], response[-1])
                )

            formatted_prompts = [
                self.format_input(prompt) for prompt in formatted_prompts
            ]

            generated_responses = self.llm.generate(
                formatted_prompts,
                **self.llm.generation_kwargs,  # type: ignore
            )

            if self.store_evolutions:
                responses = [
                    response + [evolved_response[0]]
                    for response, evolved_response in zip(
                        responses, generated_responses
                    )
                ]
            else:
                responses = [
                    [evolved_response[0]] for evolved_response in generated_responses
                ]

            self._logger.info(
                f"🔄 Ran iteration {iter_no} evolving {len(responses)} responses!"
            )

        return responses

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list of Python dictionaries with the outputs of the task.
        """

        responses = self._evolve_reponses(inputs)

        if self.store_evolutions:
            # Remove the input instruction from the `evolved_responses` list
            from_ = 1 if not self.include_original_response else 0
            responses = [response[from_:] for response in responses]

        for input, response in zip(inputs, responses):
            input.update(self.format_output(response))
        yield inputs

        self._logger.info(f"🎉 Finished evolving {len(responses)} instructions!")
