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

from datasets import load_dataset
from distilabel.llm import AnyscaleLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask

if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[10:12]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )
    pipe = Pipeline(
        generator=AnyscaleLLM(
            model="HuggingFaceH4/zephyr-7b-beta",
            task=TextGenerationTask(),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    )
    new_dataset = pipe.generate(
        dataset,  # type: ignore
        num_generations=1,
    )

    # Push to the HuggingFace Hub
    new_dataset.push_to_hub(
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
        rg_dataset = new_dataset.to_argilla()
        rg_dataset.push_to_argilla(
            name=f"my-dataset-{uuid4()}",
            workspace="admin",
        )
    except ImportError:
        pass
