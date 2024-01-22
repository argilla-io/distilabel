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
import time

from datasets import concatenate_datasets, load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import EvolInstructTask

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

    # In the original paper there is defined a number of evolutions M that determines
    # the number of evolutions that will be used for each instruction. We will use M=2
    # The most direct way of obtaining this is to concatenate the dataset M times.
    M = 4
    dataset = concatenate_datasets([dataset] * M).shuffle()

    # Let's define the pipeline using OpenAI ChatGPT and the EvolInstruct task
    pipe = Pipeline(
        generator=OpenAILLM(
            task=EvolInstructTask(),
            openai_api_key=os.getenv("OPENAI_API_KEY", None),
            num_threads=4,
            max_new_tokens=1024,
        )
    )

    start = time.time()
    ds_evol_instruct_alpaca = pipe.generate(dataset, batch_size=8)
    end = time.time()
    print("Elapsed", end - start)

    # Push to the HuggingFace Hub
    ds_evol_instruct_alpaca.push_to_hub(
        HF_REPO_ID,  # type: ignore
        split="train",
        private=False,
        token=os.getenv("HF_TOKEN", None),
    )

    try:
        import argilla as rg

        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY"),
        )

        # Convert into an Argilla dataset and push it to Argilla
        rg_dataset = ds_evol_instruct_alpaca.to_argilla()
        rg_dataset.push_to_argilla(
            name="distilabel-sample-evol-instruct",
            workspace="admin",
        )
    except ImportError:
        pass
