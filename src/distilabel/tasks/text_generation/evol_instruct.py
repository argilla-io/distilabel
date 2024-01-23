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

import random
import string
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, get_args

from distilabel.logger import get_logger
from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.utils.argilla import (
    infer_fields_from_dataset_row,
    model_metadata_from_dataset_row,
)
from distilabel.utils.imports import _ARGILLA_AVAILABLE

if _ARGILLA_AVAILABLE:
    import argilla as rg

if TYPE_CHECKING:
    from argilla import FeedbackDataset, FeedbackRecord

logger = get_logger()


_EVOL_INSTRUCT_TEMPLATE = get_template("evol-instruct.jinja2")


EvolutionMethod = Literal[
    "breadth", "constraints", "deepen", "concretizing", "reasoning"
]


def _get_stopwords() -> List[str]:
    """Gets the list of english stopwords from nltk package.

    Returns:
        List[str]: stopwords list.
    """
    try:
        with (
            importlib_resources.files("distilabel") / "tasks/_internal/stopwords_en.txt"
        ).open("r") as f:
            return f.read().split("\n")
    except FileNotFoundError:
        return []


@dataclass
class EvolInstructTask(TextGenerationTask):
    """A `TextGenerationTask` following the `EvolInstruct` specification for building the prompts.

    From the reference repository: *Evol-Instruct is a novel method using LLMs instead of humans to automatically mass-produce
    open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs.*

    The task is defined as follows:
    Starting from an initial (simpler) instruction, select in-depth or in-breadth evolving to upgrade the simple instruction
    to a more complex one or create a new one (to increase diversity).
    The In-depth Evolving includes the following operations: add constraints, deepening, concretizing and increase reasoning.
    The In-breadth Evolving is mutation, i.e., generating a completely new instruction based on the given instruction.

    Given the evolved instructions are generated from LLMs, sometimes the evolving will fail. We adopt an instruction eliminator
    to filter the failed instructions, called Elimination Evolving, but we don't apply the step of asking again to the LLM it the
    answer is a copy from the same used prompt.

    This evolutionary process can be repeated for several rounds to obtain instruction data containing various complexities.
    Currently the task is implemented as a single step, so to generate multiple evolutions you can "repeat" the instructions
    in the original dataset. An example can be seen at the following script:
    [examples/pipeline-evol-instruct-alpaca.py](https://github.com/argilla-io/distilabel/tree/main/examples/pipeline-evol-instruct-alpaca.py)

    Args:
        system_prompt (str, optional): the system prompt to be used. Not defined for this task.

    References:
        - [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/abs/2304.12244)
    """

    system_prompt: str = ""

    __jinja2_template__: str = _EVOL_INSTRUCT_TEMPLATE

    def generate_prompt(
        self, input: str, evolution_method: Optional[EvolutionMethod] = None, **_: Any
    ) -> Prompt:
        """Generates a prompt following the Evol-Instruct specification.

        Args:
            input (str): the input to be used for the prompt.
            evolution_method (str, optional): The evolution method to be used. If not provided (the default), a random one is chosen
                like the original paper. Available ones are "breadth", "constraints", "deepen", "concretizing" and "reasoning".

        Returns:
            Prompt: the generated prompt.

        Examples:
            >>> from distilabel.tasks.text_generation import EvolInstructTask
            >>> task = EvolInstructTask()
            >>> task.generate_prompt("Give three tips for staying healthy.")
            Prompt(
                system_prompt="",
                formatted_prompt="I want you to act as a Prompt ...",
            )
        """
        if not evolution_method:
            evolution_method = random.choice(get_args(EvolutionMethod))

        render_kwargs = {
            "evol_method": evolution_method,
            "instruction": input,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    @property
    def output_args_names(self) -> List[str]:
        return ["instruction"]

    def _elimination_evolving(self, output: str) -> Optional[str]:
        """Performs the elimination step of the Evol-Instruct task, steps 2-4 in the paper:

        1. [NOT IMPLEMENTED] The evolved instruction does not provide any information gain compared
        to the original one. Use ChatGPT to make this determination, this is outlined in Appendix G of the original paper.
        2. The evolved instruction makes it difficult for the LLM to generate a response. We found that
        when the generated response contains “sorry” and is relatively short in length (i.e., less than
        80 words), it often indicates that the LLM struggles to respond to the evolved instruction.
        So we can use this rule to make a judgment.
        3. The response generated by the LLM only contains punctuation and stop words.
        4. The evolved instruction obviously copies some words from the evolving prompt, such as
        “given prompt”, “rewritten prompt”, “#Rewritten Prompt#”, etc.
        """
        output = output.strip()
        if output == "":
            return

        # 2) The evolved instruction makes it difficult for the LLM to generate a response.
        if "sorry" in output.lower() and len(output.split(" ")) < 80:
            logger.info(
                f"Evolution step removed the output, it's hard for the LLM to generate a response: {output}"
            )
            return

        # 3) The output only contains punctuation and stop words
        stopwords = _get_stopwords()
        clean_output = [word for word in output.split(" ") if word not in stopwords]
        if set(clean_output).difference(set(string.punctuation)) == 0:
            logger.info(
                f"Evolution step removed the output, it only contains punctuation and stop words: {output}"
            )
            return

        # 4) Remove copied words from the prompt
        prompt_words = {
            "#Given Prompt#",
            "#Created Prompt#",
            "given prompt",
            "created prompt",
            "#The Given Prompt#",
            "#Rewritten Prompt#",
            "rewritten prompt",
        }
        if any(word in output for word in prompt_words):
            logger.info(
                f"Evolution step removed the output due to word repetition from the prompt: {output}"
            )
            return

        return output

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the output of the model into the desired format, applying the elimination step for bad generations.

        Args:
            output (str): the output of the model.

        Note:
            The eliminatin step is applied to the output, but only steps 2-4 in the paper are implemented.
            Refer to point 3.2, Elimination Evolving section in [`WizardLM: Empowering Large Language Models to Follow Complex Instructions`](https://arxiv.org/abs/2304.12244)
            for more information on the elimination evolving step, and take a look at the `_elimination_evolving`
            method for more information of the implementation.
        """
        output = self._elimination_evolving(output)
        return {"instruction": output}

    def to_argilla_dataset(self, dataset_row: Dict[str, Any]) -> "FeedbackDataset":
        # First we infer the fields from the input_args_names, but we could also
        # create those manually instead using `rg.TextField(...)`
        fields = infer_fields_from_dataset_row(
            field_names=self.input_args_names,
            dataset_row=dataset_row,
        )
        # Once the input fields have been defined, then we also include the instruction
        # field which will be fulfilled with each of the instructions generated.
        fields.append(rg.TextField(name="instruction", title="instruction"))  # type: ignore
        # Then we add a default `RatingQuestion` which asks the users to provide a
        # rating for each of the generations, differing from the scenario where the inputs
        # are the fields and the outputs the ones used to formulate the quesstions. So on,
        # in this scenario we won't have suggestions, as the questions will be related to the
        # combination of inputs and outputs.
        questions = [
            rg.RatingQuestion(  # type: ignore
                name="instruction-rating",
                title="How would you rate the generated instruction?",
                values=list(range(1, 11)),
            )
        ]
        # Finally, we define some metadata properties that can be potentially used
        # while exploring the dataset within Argilla to get more insights on the data.
        metadata_properties = []
        for arg_name in self.input_args_names:
            if isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    metadata_properties.append(
                        rg.IntegerMetadataProperty(name=f"length-{arg_name}-{idx}")  # type: ignore
                    )
            elif isinstance(dataset_row[arg_name], str):
                metadata_properties.append(
                    rg.IntegerMetadataProperty(name=f"length-{arg_name}")  # type: ignore
                )
            else:
                warnings.warn(
                    f"Unsupported input type ({type(dataset_row[arg_name])}), skipping...",
                    UserWarning,
                    stacklevel=2,
                )
        metadata_properties.append(
            rg.IntegerMetadataProperty(name="length-instruction")  # type: ignore
        )  # type: ignore
        # Then we just return the `FeedbackDataset` with the fields, questions, and metadata properties
        # defined above.
        return rg.FeedbackDataset(
            fields=fields,
            questions=questions,  # type: ignore
            metadata_properties=metadata_properties,  # Note that these are always optional
        )

    def to_argilla_record(
        self,
        dataset_row: Dict[str, Any],
        instructions_column: Optional[str] = "instruction",
    ) -> List["FeedbackRecord"]:
        """Converts a dataset row to a list of Argilla `FeedbackRecord`s."""
        records = []
        for instruction in dataset_row[instructions_column]:  # type: ignore
            fields, metadata = {}, {}
            for arg_name in self.input_args_names:
                arg_value = dataset_row[arg_name]
                if isinstance(arg_value, list):
                    for idx, value in enumerate(arg_value, start=1):
                        value = value.strip() if isinstance(value, str) else ""
                        fields[f"{arg_name}-{idx}"] = value
                        if value is not None:
                            metadata[f"length-{arg_name}-{idx}"] = len(value)
                elif isinstance(arg_value, str):
                    fields[arg_name] = arg_value.strip() if arg_value else ""
                    if arg_value is not None:
                        metadata[f"length-{arg_name}"] = len(arg_value.strip())
                else:
                    warnings.warn(
                        f"Unsupported input type ({type(arg_value)}), skipping...",
                        UserWarning,
                        stacklevel=2,
                    )
            fields["instruction"] = instruction
            metadata["length-instruction"] = len(instruction)

            # Then we add the model metadata from the `generation_model` and `labelling_model`
            # columns of the dataset, if they exist.
            metadata.update(model_metadata_from_dataset_row(dataset_row=dataset_row))
            # Finally, we append the `FeedbackRecord` with the fields and the metadata
            records.append(rg.FeedbackRecord(fields=fields, metadata=metadata))
        if not records:
            raise ValueError(
                f"Skipping the row {dataset_row} as the list of `FeedbackRecord` is empty as those could not be inferred."
            )
        return records
