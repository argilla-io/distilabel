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

from datasets import Dataset
from distilabel.llm import TogetherInferenceLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask

if __name__ == "__main__":
    dataset = Dataset.from_dict(
        {
            "input": ["Explain me the theory of relativity as if you were a pirate."],
        }
    )

    llm = TogetherInferenceLLM(
        model="togethercomputer/llama-2-70b-chat",
        api_key=os.getenv("TOGETHER_API_KEY", None),
        task=TextGenerationTask(),
        prompt_format="llama2",
    )
    pipeline = Pipeline(generator=llm)

    start = time.time()
    dataset = pipeline.generate(
        dataset=dataset,
        shuffle_before_labelling=False,
        num_generations=2,
        skip_dry_run=True,
        display_progress_bar=False,
    )  # type: ignore
    end = time.time()
    print("Elapsed", end - start)

    # Push to the HuggingFace Hub
    dataset.push_to_hub(
        os.getenv("HF_REPO_ID"),  # type: ignore
        split="train",
        private=True,
        token=os.getenv("HF_TOKEN", None),
    )

    try:
        from uuid import uuid4

        import argilla as rg

        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"),
            api_key=os.getenv("ARGILLA_API_KEY"),
        )

        # Convert into an Argilla dataset and push it to Argilla
        rg_dataset = dataset.to_argilla()
        rg_dataset.push_to_argilla(
            name=f"my-dataset-{uuid4()}",
            workspace="admin",
        )
    except ImportError:
        pass
