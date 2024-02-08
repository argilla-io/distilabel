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
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from datasets import Dataset, load_dataset
from distilabel.dataset import CustomDataset
from distilabel.llm import LLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import EvolInstructTask, Prompt, TextGenerationTask

HF_REPO_ID = "argilla/distilabel-sample-evol-instruct"


if __name__ == "__main__":
    # For this example we will use the Alpaca dataset directly from the WizardLM repo
    ds_alpaca = load_dataset(
        "json",
        data_files=r"https://raw.githubusercontent.com/nlpxucan/WizardLM/main/training/data/alpaca_data.json",
        split="train",
    )

    # We will need as a column input with the initial instruction so will prepare the data accordingly
    def prepare(example):
        return {
            "input": f"{example['instruction'].strip()}\n{example['input'].strip()}"
        }

    ds_alpaca = ds_alpaca.map(prepare, remove_columns=ds_alpaca.column_names)
    # Select a subset of the dataset for this example
    dataset = ds_alpaca.shuffle().select(range(5))

    # Define our LLM for the evol-instruct task
    evolver_llm = OpenAILLM(
        task=EvolInstructTask(),
        api_key=os.getenv("OPENAI_API_KEY"),
        num_threads=4,
        max_new_tokens=1024,
    )

    # The paper defines 4 steps for the elimination stage. The last three are implemented in the EvolInstructTask.
    # For the first one (elimination of the equal prompts), we will create a custom task.
    elimination_equal_prompt = """Here are two Instructions, do you think they are equal to each other and meet the following requirements?:
        1. They have the same constraints and requirements.
        2. They have the same depth and breadth of the inquiry.
        The First Prompt: {first_instruction}
        The Second Prompt: {second_instruction}
        Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason):"""

    @dataclass
    class EliminationEqualPrompts(TextGenerationTask):
        """Task to check for the equality of two instructions following the Appendix G in
        [WizardLM paper](https://arxiv.org/abs/2304.12244).
        """

        system_prompt: str = "You are an AI judge in charge of determining the equality of two instructions."

        def generate_prompt(self, input: List[str]) -> Prompt:
            return Prompt(
                system_prompt=self.system_prompt,
                formatted_prompt=elimination_equal_prompt.format(
                    first_instruction=input[0], second_instruction=input[1]
                ),
            )

        def parse_output(self, output: str) -> List[Dict[str, str]]:
            """Remove punctuation from the string and lowercase it."""
            return {
                "generations": output.translate(
                    str.maketrans("", "", string.punctuation)
                ).lower()
            }

    # Helper function to prepare the dataset for the elimination task with the original
    # instructions and the evolved ones. If the evolved instruction is None,
    # we use the original instruction (to make sure it will be removed)
    def prepare_for_equal_prompts(example):
        if example["instructions"][0] is None:
            return {"input": [example["input"], example["input"]]}
        else:
            return {"input": [example["input"], example["instructions"][0]]}

    # Define our LLM for the elimination task
    elimination_llm = OpenAILLM(
        task=EliminationEqualPrompts(),
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        num_threads=4,
        max_new_tokens=2048,
    )

    # Similarly to the paper, we will create a function to iterate M times over the dataset
    # generating new instructions and then using the elimination task to remove the equal ones.
    start = time.time()

    def make_evol_instruct_dataset(
        evolver_llm: LLM,
        elimination_llm: LLM,
        dataset: Dataset,
        evolution_steps: int = 4,
    ) -> "Dataset":
        # Set the pipelines
        evolver_pipe = Pipeline(generator=evolver_llm)
        elimination_pipe = Pipeline(generator=elimination_llm)

        # Set the initial dataset
        input_dataset = dataset
        successful_instructions = []

        # Start the evolution process
        for step in range(1, evolution_steps + 1):
            print(f"Evolving dataset step: {step}/{evolution_steps}")

            # Generate new instructions
            evolved_dataset = evolver_pipe.generate(input_dataset, batch_size=8)

            # Prepare the dataset for the elimination process
            prepared_dataset = evolved_dataset.map(
                prepare_for_equal_prompts
            ).select_columns(["input"])

            # Perform the elimination process, step 1
            elimination_dataset = elimination_pipe.generate(
                prepared_dataset, batch_size=8
            )

            # Save the successful instructions in the pool and prepare the inputs for the next iteration
            new_instructions = []

            for row_evolved, row_elimination in zip(
                evolved_dataset, elimination_dataset
            ):
                if (row_evolved["instructions"][0] is not None) and (
                    row_elimination["generations"][0] != "equal"
                ):
                    new_instructions.append(row_evolved["instructions"][0])
                    successful_instructions.append(row_evolved)
                else:
                    new_instructions.append(row_evolved["input"])

            input_dataset = Dataset.from_dict({"input": new_instructions})

        # Prepare the final dataset
        df_final_dataset = pd.DataFrame(successful_instructions)
        final_dataset = Dataset.from_pandas(df_final_dataset)
        final_dataset.__class__ = CustomDataset
        final_dataset.task = EvolInstructTask()

        return final_dataset

    ds_evol_instruct = make_evol_instruct_dataset(
        evolver_llm=evolver_llm,
        elimination_llm=elimination_llm,
        dataset=dataset,
        evolution_steps=4,
    )

    end = time.time()
    print("Elapsed", end - start)

    # Push to the HuggingFace Hub
    ds_evol_instruct.push_to_hub(
        HF_REPO_ID,  # type: ignore
        split="train",
        private=False,
        token=os.getenv("HF_TOKEN", None),
    )

    # Push to Argilla
    try:
        import argilla as rg

        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY"),
        )

        # Convert into an Argilla dataset and push it to Argilla
        rg_dataset = ds_evol_instruct.to_argilla()
        rg_dataset.push_to_argilla(
            name="distilabel-sample-evol-instruct",
            workspace="admin",
        )
    except ImportError:
        pass
