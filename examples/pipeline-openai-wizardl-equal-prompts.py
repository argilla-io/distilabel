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

import os
import string
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Prompt, TextGenerationTask

if __name__ == "__main__":
    ds = load_dataset("argilla/distilabel-sample-evol-instruct", split="train")

    # Map the sample dataset to contain in the input both the original instruction and the
    # Evol Instructed generation:
    def prepare_to_for_equal_prompts(example):
        return {"input": [example["input"], example["instructions"][0]]}

    # Prepare the dataset and remove the previous columns but the "input" one.
    new_ds = (
        ds.select(range(4))
        .map(prepare_to_for_equal_prompts)
        .remove_columns(set(ds.column_names) - {"input"})
    )

    # Prompt from the WizardLM paper for the Equal Prompts task:
    wizardllm_equal_prompt = """Here are two Instructions, do you think they are equal to each other and meet the following requirements?:
    1. They have the same constraints and requirments.
    2. They have the same depth and breadth of the inquiry.
    The First Prompt: {first_instruction}
    The Second Prompt: {second_instruction}
    Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason):"""

    @dataclass
    class WizardLMEqualPrompts(TextGenerationTask):
        """Task to check for the equality of two instructions following the Appendix G in
        [WizardLM paper](https://arxiv.org/abs/2304.12244).
        """

        system_prompt: str = "You are an AI judge in charge of determining the equality of two instructions. "

        def generate_prompt(self, input: List[str]) -> Prompt:
            return Prompt(
                system_prompt=self.system_prompt,
                formatted_prompt=wizardllm_equal_prompt.format(
                    first_instruction=input[0], second_instruction=input[1]
                ),
            )

        def parse_output(self, output: str) -> List[Dict[str, str]]:
            """Remove punctuation from the string."""
            return {
                "generations": output.translate(
                    str.maketrans("", "", string.punctuation)
                )
            }

    # Define a generator pipeline with gpt-3.5-turbo as in the original paper
    pipe = Pipeline(
        generator=OpenAILLM(
            task=WizardLMEqualPrompts(),
            openai_api_key=os.getenv("OPENAI_API_KEY", None),
            temperature=0.3,
        )
    )

    # Run the pipeline, and afterwards print a couple of examples to see the results
    ds_equals = pipe.generate(new_ds, batch_size=4)

    # >>> from rich import print as rprint
    # >>> rprint(ds_equals.select_columns(["input", "generations"])[:2])
    # {
    #     'input': [
    #         [
    #             'Create a sentence using the words "happy," "joyful," and "thrilled."\n',
    #             'Compose a concise and articulate sentence that incorporates the terms "ecstatic," "exhilarated,"
    # "blissful," and "overjoyed."'
    #         ],
    #         [
    #             'Construct plumbing diagrams for a two-story house\n',
    #             'Design comprehensive and detailed plumbing diagrams for a two-story house that include separate
    # diagrams for each floor, showcasing the layout, dimensions, and connections of all plumbing fixtures, pipelines,
    # drains, vents, water supply sources, and sewage disposal systems. These diagrams should consider the specific
    # requirements and codes of local building regulations while providing a clear and accurate representation of the
    # plumbing infrastructure throughout the entire house.'
    #         ]
    #     ],
    #     'generations': [['Equal'], ['Not Equal']]
    # }
