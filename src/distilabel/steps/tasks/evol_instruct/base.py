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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from pydantic import Field
from typing_extensions import override

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import StepInput
from distilabel.steps.tasks.base import Task
from distilabel.steps.tasks.evol_instruct.utils import MUTATION_TEMPLATES
from distilabel.steps.tasks.typing import ChatType
from distilabel.utils.lists import flatten_responses

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class EvolInstruct(Task):
    """Evolve instructions using an `LLM`.

    WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Attributes:
        num_evolutions: The number of evolutions to be performed.
        store_evolutions: Whether to store all the evolutions or just the last one. Defaults
            to `False`.
        generate_answers: Whether to generate answers for the evolved instructions. Defaults
            to `False`.
        include_original_instruction: Whether to include the original instruction in the
            `evolved_instructions` output column. Defaults to `False`.
        mutation_templates: The mutation templates to be used for evolving the instructions.
            Defaults to the ones provided in the `utils.py` file.
        seed: The seed to be set for `numpy` in order to randomly pick a mutation method.
            Defaults to `42`.

    Runtime parameters:
        - `seed`: The seed to be set for `numpy` in order to randomly pick a mutation method.

    Input columns:
        - instruction (`str`): The instruction to evolve.

    Output columns:
        - evolved_instruction (`str`): The evolved instruction if `store_evolutions=False`.
        - evolved_instructions (`List[str]`): The evolved instructions if `store_evolutions=True`.
        - model_name (`str`): The name of the LLM used to evolve the instructions.
        - answer (`str`): The answer to the evolved instruction if `generate_answers=True`
            and `store_evolutions=False`.
        - answers (`List[str]`): The answers to the evolved instructions if `generate_answers=True`
            and `store_evolutions=True`.

    Categories:
        - evol
        - instruction

    References:
        - [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
        - [GitHub: h2oai/h2o-wizardlm](https://github.com/h2oai/h2o-wizardlm)

    Examples:

        Evolve an instruction using an LLM:

        ```python
        from distilabel.steps.tasks import EvolInstruct
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        evol_instruct = EvolInstruct(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            num_evolutions=2,
        )

        evol_instruct.load()

        result = next(evol_instruct.process([{"instruction": "common instruction"}]))
        # result
        # [{'instruction': 'common instruction', 'evolved_instruction': 'evolved instruction', 'model_name': 'model_name'}]
        ```

        Keep the iterations of the evolutions:

        ```python
        from distilabel.steps.tasks import EvolInstruct
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        evol_instruct = EvolInstruct(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            num_evolutions=2,
            store_evolutions=True,
        )

        evol_instruct.load()

        result = next(evol_instruct.process([{"instruction": "common instruction"}]))
        # result
        # [
        #     {
        #         'instruction': 'common instruction',
        #         'evolved_instructions': ['initial evolution', 'final evolution'],
        #         'model_name': 'model_name'
        #     }
        # ]
        ```

        Generate answers for the instructions in a single step:

        ```python
        from distilabel.steps.tasks import EvolInstruct
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        evol_instruct = EvolInstruct(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            num_evolutions=2,
            generate_answers=True,
        )

        evol_instruct.load()

        result = next(evol_instruct.process([{"instruction": "common instruction"}]))
        # result
        # [
        #     {
        #         'instruction': 'common instruction',
        #         'evolved_instruction': 'evolved instruction',
        #         'answer': 'answer to the instruction',
        #         'model_name': 'model_name'
        #     }
        # ]
        ```

    Citations:

        ```
        @misc{xu2023wizardlmempoweringlargelanguage,
            title={WizardLM: Empowering Large Language Models to Follow Complex Instructions},
            author={Can Xu and Qingfeng Sun and Kai Zheng and Xiubo Geng and Pu Zhao and Jiazhan Feng and Chongyang Tao and Daxin Jiang},
            year={2023},
            eprint={2304.12244},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2304.12244},
        }
        ```
    """

    num_evolutions: int
    store_evolutions: bool = False
    generate_answers: bool = False
    include_original_instruction: bool = False
    mutation_templates: Dict[str, str] = MUTATION_TEMPLATES

    seed: RuntimeParameter[int] = Field(
        default=42,
        description="As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
    )

    inputs: List[str] = Field(
        default=["instruction"],
        description="The inputs for the task are the 'instruction'.",
    )
    outputs: List[str] = Field(
        default=["instruction", "model_name", "evolved_instructions", "answers"],
        description="The output for the task are 'instruction', and 'model_name'. Optionally, if `generate_answers=True` 'answers' is added.",
    )

    @override
    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

        self.outputs = (
            self.outputs
            if self.generate_answers
            else ["instruction", "model_name", "evolved_instructions"]
        )

    def format_input(self, input: str) -> ChatType:  # type: ignore
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation. And the
        `system_prompt` is added as the first message if it exists."""
        return [{"role": "user", "content": input}]

    @override
    def format_output(  # type: ignore
        self, instructions: Union[str, List[str]], answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:  # type: ignore
        """The output for the task is a dict with `evolved_instructions`, `answers` if `generate_answers=True`; and, finally, the `model_name`.

        Args:
            instructions: The instructions to be included within the output.
            answers: The answers to be included within the output if `generate_answers=True`.

        Returns:
            if `generate_answers=True` return {"evolved_instructions": ..., "model_name": ..., "answers": ...};
            if `generate_answers=False` return {"evolved_instruction": ..., "model_name": ...};
        """
        _output = {}
        if not self.store_evolutions:
            _output["evolved_instructions"] = instructions[-1]
        else:
            _output["evolved_instructions"] = instructions

        if self.generate_answers and answers:
            if not self.store_evolutions:
                _output["answers"] = answers[-1]
            else:
                _output["answers"] = answers

        _output["model_name"] = self.llm.model_name
        return _output

    @property
    def mutation_templates_names(self) -> List[str]:
        """Returns the names i.e. keys of the provided `mutation_templates`."""
        return list(self.mutation_templates.keys())

    def _apply_random_mutation(self, instruction: str) -> str:
        """Applies a random mutation from the ones provided as part of the `mutation_templates`
        enum, and returns the provided instruction within the mutation prompt.

        Args:
            instruction: The instruction to be included within the mutation prompt.

        Returns:
            A random mutation prompt with the provided instruction.
        """
        mutation = np.random.choice(self.mutation_templates_names)
        return self.mutation_templates[mutation].replace("<PROMPT>", instruction)  # type: ignore

    def _evolve_instructions(self, inputs: "StepInput") -> List[List[str]]:
        """Evolves the instructions provided as part of the inputs of the task.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Returns:
            A list where each item is a list with either the last evolved instruction if
            `store_evolutions=False` or all the evolved instructions if `store_evolutions=True`.
        """

        instructions: List[List[str]] = [[input["instruction"]] for input in inputs]

        for iter_no in range(self.num_evolutions):
            formatted_prompts = []
            for instruction in instructions:
                formatted_prompts.append(self._apply_random_mutation(instruction[-1]))

            formatted_prompts = [
                self.format_input(prompt) for prompt in formatted_prompts
            ]
            generated_prompts = flatten_responses(
                self.llm.generate(
                    formatted_prompts,
                    **self.llm.generation_kwargs,  # type: ignore
                )
            )

            evolved_instructions = []
            for generated_prompt in generated_prompts:
                generated_prompt = generated_prompt.split("Prompt#:")[-1].strip()
                evolved_instructions.append(generated_prompt)

            if self.store_evolutions:
                instructions = [
                    instruction + [evolved_instruction]
                    for instruction, evolved_instruction in zip(
                        instructions, evolved_instructions
                    )
                ]
            else:
                instructions = [
                    [evolved_instruction]
                    for evolved_instruction in evolved_instructions
                ]

            self._logger.info(
                f"🔄 Ran iteration {iter_no} evolving {len(instructions)} instructions!"
            )

        return instructions

    def _generate_answers(
        self, evolved_instructions: List[List[str]]
    ) -> List[List[str]]:
        """Generates the answer for the instructions in `instructions`.

        Args:
            evolved_instructions: A list of lists where each item is a list with either the last
                evolved instruction if `store_evolutions=False` or all the evolved instructions
                if `store_evolutions=True`.

        Returns:
            A list of answers for each instruction.
        """
        formatted_instructions = [
            self.format_input(instruction)
            for instructions in evolved_instructions
            for instruction in instructions
        ]

        responses = self.llm.generate(
            formatted_instructions,
            num_generations=1,
            **self.llm.generation_kwargs,  # type: ignore
        )

        step = (
            self.num_evolutions
            if not self.include_original_instruction
            else self.num_evolutions + 1
        )
        return [
            flatten_responses(responses[i : i + step])
            for i in range(0, len(responses), step)
        ]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            inputs: A list of Python dictionaries with the inputs of the task.

        Yields:
            A list of Python dictionaries with the outputs of the task.
        """

        evolved_instructions = self._evolve_instructions(inputs)

        if self.store_evolutions:
            # Remove the input instruction from the `evolved_instructions` list
            from_ = 1 if not self.include_original_instruction else 0
            evolved_instructions = [
                instruction[from_:] for instruction in evolved_instructions
            ]

        if not self.generate_answers:
            for input, instruction in zip(inputs, evolved_instructions):
                input.update(self.format_output(instruction))
            yield inputs

        self._logger.info(
            f"🎉 Finished evolving {len(evolved_instructions)} instructions!"
        )

        if self.generate_answers:
            self._logger.info(
                f"🧠 Generating answers for the {len(evolved_instructions)} evolved instructions!"
            )

            answers = self._generate_answers(evolved_instructions)

            self._logger.info(
                f"🎉 Finished generating answers for the {len(evolved_instructions)} evolved"
                " instructions!"
            )

            for idx, (input, instruction) in enumerate(
                zip(inputs, evolved_instructions)
            ):
                input.update(self.format_output(instruction, answers[idx]))
            yield inputs
