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

import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from collections import defaultdict
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field, PrivateAttr
from typing_extensions import override

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.tasks.base import GeneratorTask
from distilabel.steps.tasks.evol_instruct.utils import GENERATION_MUTATION_TEMPLATES
from distilabel.utils.lists import flatten_responses

if TYPE_CHECKING:
    from distilabel.llms.typing import LLMStatistics
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import GeneratorStepOutput


class EvolInstructGenerator(GeneratorTask):
    """Generate evolved instructions using an `LLM`.

    WizardLM: Empowering Large Language Models to Follow Complex Instructions

    Attributes:
        num_instructions: The number of instructions to be generated.
        generate_answers: Whether to generate answers for the instructions or not. Defaults
            to `False`.
        mutation_templates: The mutation templates to be used for the generation of the
            instructions.
        min_length: Defines the length (in bytes) that the generated instruction needs to
            be higher than, to be considered valid. Defaults to `512`.
        max_length: Defines the length (in bytes) that the generated instruction needs to
            be lower than, to be considered valid. Defaults to `1024`.
        seed: The seed to be set for `numpy` in order to randomly pick a mutation method.
            Defaults to `42`.

    Runtime parameters:
        - `min_length`: Defines the length (in bytes) that the generated instruction needs
            to be higher than, to be considered valid.
        - `max_length`: Defines the length (in bytes) that the generated instruction needs
            to be lower than, to be considered valid.
        - `seed`: The seed to be set for `numpy` in order to randomly pick a mutation method.

    Output columns:
        - instruction (`str`): The generated instruction if `generate_answers=False`.
        - answer (`str`): The generated answer if `generate_answers=True`.
        - instructions (`List[str]`): The generated instructions if `generate_answers=True`.
        - model_name (`str`): The name of the LLM used to generate and evolve the instructions.

    Categories:
        - evol
        - instruction
        - generation

    References:
        - [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
        - [GitHub: h2oai/h2o-wizardlm](https://github.com/h2oai/h2o-wizardlm)

    Examples:
        Generate evolved instructions without initial instructions:

        ```python
        from distilabel.steps.tasks import EvolInstructGenerator
        from distilabel.models import InferenceEndpointsLLM

        # Consider this as a placeholder for your actual LLM.
        evol_instruct_generator = EvolInstructGenerator(
            llm=InferenceEndpointsLLM(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
            ),
            num_instructions=2,
        )

        evol_instruct_generator.load()

        result = next(scorer.process())
        # result
        # [{'instruction': 'generated instruction', 'model_name': 'test'}]
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

    num_instructions: int
    generate_answers: bool = False
    mutation_templates: Dict[str, str] = GENERATION_MUTATION_TEMPLATES

    min_length: RuntimeParameter[int] = Field(
        default=512,
        description="Defines the length (in bytes) that the generated instruction needs to be higher than, to be considered valid.",
    )
    max_length: RuntimeParameter[int] = Field(
        default=1024,
        description="Defines the length (in bytes) that the generated instruction needs to be lower than, to be considered valid.",
    )

    seed: RuntimeParameter[int] = Field(
        default=42,
        description="As `numpy` is being used in order to randomly pick a mutation method, then is nice to seed a random seed.",
    )
    _seed_texts: Optional[List[str]] = PrivateAttr(default_factory=list)
    _prompts: Optional[List[str]] = PrivateAttr(default_factory=list)

    def _generate_seed_texts(self) -> List[str]:
        """Generates a list of seed texts to be used as part of the starting prompts for the task.

        It will use the `FRESH_START` mutation template, as it needs to generate text from scratch; and
        a list of English words will be used to generate the seed texts that will be provided to the
        mutation method and included within the prompt.

        Returns:
            A list of seed texts to be used as part of the starting prompts for the task.
        """
        seed_texts = []
        for _ in range(self.num_instructions * 10):
            num_words = np.random.choice([1, 2, 3, 4])
            seed_texts.append(
                self.mutation_templates["FRESH_START"].replace(  # type: ignore
                    "<PROMPT>",
                    ", ".join(
                        [
                            np.random.choice(self._english_nouns).strip()
                            for _ in range(num_words)
                        ]
                    ),
                )
            )
        return seed_texts

    @override
    def model_post_init(self, __context: Any) -> None:
        """Override this method to perform additional initialization after `__init__` and `model_construct`.
        This is useful if you want to do some validation that requires the entire model to be initialized.
        """
        super().model_post_init(__context)

        np.random.seed(self.seed)

        self._seed_texts = self._generate_seed_texts()
        self._prompts = [
            np.random.choice(self._seed_texts) for _ in range(self.num_instructions)
        ]

    @cached_property
    def _english_nouns(self) -> List[str]:
        """A list of English nouns to be used as part of the starting prompts for the task.

        References:
            - https://github.com/h2oai/h2o-wizardlm
        """
        _path = str(
            importlib_resources.files("distilabel")
            / "steps/tasks/evol_instruct/english_nouns.txt"
        )
        with open(_path, mode="r") as f:
            return [line.strip() for line in f.readlines()]

    @property
    def outputs(self) -> List[str]:
        """The output for the task are the `instruction`, the `answer` if `generate_answers=True`
        and the `model_name`."""
        _outputs = ["instruction", "model_name"]
        if self.generate_answers:
            _outputs.append("answer")
        return _outputs

    def format_output(  # type: ignore
        self, instruction: str, answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """The output for the task is a dict with: `instruction`; `answer` if `generate_answers=True`;
        and, finally, the `model_name`.

        Args:
            instruction: The instruction to be included within the output.
            answer: The answer to be included within the output if `generate_answers=True`.

        Returns:
            If `generate_answers=True` return {"instruction": ..., "answer": ..., "model_name": ...};
            if `generate_answers=False` return {"instruction": ..., "model_name": ...};
        """
        _output = {
            "instruction": instruction,
            "model_name": self.llm.model_name,
        }
        if self.generate_answers and answer is not None:
            _output["answer"] = answer
        return _output

    @property
    def mutation_templates_names(self) -> List[str]:
        """Returns the names i.e. keys of the provided `mutation_templates`."""
        return list(self.mutation_templates.keys())

    def _apply_random_mutation(self, iter_no: int) -> List["ChatType"]:
        """Applies a random mutation from the ones provided as part of the `mutation_templates`
        enum, and returns the provided instruction within the mutation prompt.

        Args:
            iter_no: The iteration number to be used to check whether the iteration is the
                first one i.e. FRESH_START, or not.

        Returns:
            A random mutation prompt with the provided instruction formatted as an OpenAI conversation.
        """
        prompts = []
        for idx in range(self.num_instructions):
            if (
                iter_no == 0
                or "Write one question or request containing" in self._prompts[idx]  # type: ignore
            ):
                mutation = "FRESH_START"
            else:
                mutation = np.random.choice(self.mutation_templates_names)
                if mutation == "FRESH_START":
                    self._prompts[idx] = np.random.choice(self._seed_texts)  # type: ignore

            prompt_with_template = (
                self.mutation_templates[mutation].replace(  # type: ignore
                    "<PROMPT>",
                    self._prompts[idx],  # type: ignore
                )  # type: ignore
                if iter_no != 0
                else self._prompts[idx]  # type: ignore
            )
            prompts.append([{"role": "user", "content": prompt_with_template}])
        return prompts

    def _generate_answers(
        self, instructions: List[List[str]]
    ) -> Tuple[List[str], "LLMStatistics"]:
        """Generates the answer for the last instruction in `instructions`.

        Args:
            instructions: A list of lists where each item is a list with either the last
                evolved instruction if `store_evolutions=False` or all the evolved instructions
                if `store_evolutions=True`.

        Returns:
            A list of answers for the last instruction in `instructions`.
        """
        # TODO: update to generate answers for all the instructions
        _formatted_instructions = [
            [{"role": "user", "content": instruction[-1]}]
            for instruction in instructions
        ]
        responses = self.llm.generate(
            _formatted_instructions,
            **self.llm.generation_kwargs,  # type: ignore
        )
        statistics: Dict[str, Any] = defaultdict(list)
        for response in responses:
            for k, v in response["statistics"].items():
                statistics[k].append(v[0])

        return flatten_responses(
            [response["generations"] for response in responses]
        ), dict(statistics)

    @override
    def process(self, offset: int = 0) -> "GeneratorStepOutput":  # NOQA: C901, type: ignore
        """Processes the inputs of the task and generates the outputs using the LLM.

        Args:
            offset: The offset to start the generation from. Defaults to 0.

        Yields:
            A list of Python dictionaries with the outputs of the task, and a boolean
            flag indicating whether the task has finished or not i.e. is the last batch.
        """
        instructions = []
        mutation_no = 0

        # TODO: update to take into account `offset`
        iter_no = 0
        while len(instructions) < self.num_instructions:
            prompts = self._apply_random_mutation(iter_no=iter_no)

            # TODO: Update the function to extract from the dict
            responses = self.llm.generate(prompts, **self.llm.generation_kwargs)  # type: ignore

            generated_prompts = flatten_responses(
                [response["generations"] for response in responses]
            )
            statistics: "LLMStatistics" = defaultdict(list)
            for response in responses:
                for k, v in response["statistics"].items():
                    statistics[k].append(v[0])

            for idx, generated_prompt in enumerate(generated_prompts):
                generated_prompt = generated_prompt.split("Prompt#:")[-1].strip()
                if self.max_length >= len(generated_prompt) >= self.min_length:  # type: ignore
                    instructions.append(generated_prompt)
                    self._prompts[idx] = np.random.choice(self._seed_texts)  # type: ignore
                else:
                    self._prompts[idx] = generated_prompt  # type: ignore

            self._logger.info(
                f"🔄 Ran iteration {iter_no} with {len(instructions)} instructions already evolved!"
            )
            iter_no += 1

            if len(instructions) > self.num_instructions:
                instructions = instructions[: self.num_instructions]
            if len(instructions) > mutation_no:
                mutation_no = len(instructions) - mutation_no

            if not self.generate_answers and len(instructions[-mutation_no:]) > 0:
                formatted_generations = []
                for mutated_instruction in instructions[-mutation_no:]:
                    mutated_instruction = self.format_output(mutated_instruction)
                    mutated_instruction["distilabel_metadata"] = {
                        f"statistics_instruction_{self.name}": dict(statistics)
                    }
                    formatted_generations.append(mutated_instruction)
                yield (
                    formatted_generations,
                    len(instructions) >= self.num_instructions,
                )

        self._logger.info(f"🎉 Finished evolving {len(instructions)} instructions!")

        if self.generate_answers:
            self._logger.info(
                f"🧠 Generating answers for the {len(instructions)} evolved instructions!"
            )

            answers, statistics = self._generate_answers(instructions)

            self._logger.info(
                f"🎉 Finished generating answers for the {len(instructions)} evolved instructions!"
            )

            formatted_outputs = []
            for instruction, answer in zip(instructions, answers):
                formatted_output = self.format_output(instruction, answer)
                formatted_output["distilabel_metadata"] = {
                    f"statistics_answer_{self.name}": dict(statistics)
                }
                formatted_outputs.append(formatted_output)

            yield (
                formatted_outputs,
                True,
            )

    @override
    def _sample_input(self) -> "ChatType":
        return self._apply_random_mutation(iter_no=0)[0]
