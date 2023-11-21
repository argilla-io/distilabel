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

from datasets import load_dataset
from distilabel.llm import InferenceEndpointsLLM
from distilabel.pipeline import pipeline
from distilabel.tasks import Llama2TextGenerationTask

if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[:100]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )

    pipe = pipeline(
        "preference",
        "text-quality",
        generator=InferenceEndpointsLLM(
            endpoint_name=os.getenv("HF_INFERENCE_ENDPOINT_NAME"),  # type: ignore
            endpoint_namespace=os.getenv("HF_NAMESPACE", None),
            task=Llama2TextGenerationTask(),
            max_new_tokens=256,
            num_threads=2,
            temperature=0.3,
        ),
        max_new_tokens=256,
        num_threads=2,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,
    )

    start = time.time()
    dataset = pipe.generate(
        dataset,  # type: ignore
        num_generations=2,
        batch_size=1,
        enable_checkpoints=True,
        display_progress_bar=True,
    )
    end = time.time()
    print("Elapsed", end - start)

    dataset.push_to_hub(
        os.getenv("HF_REPO_ID"),  # type: ignore
        split="train",
        private=False,
    )

    try:
        from uuid import uuid4

        import argilla as rg

        rg.init(
            api_url=os.getenv("ARGILLA_API_URL"), api_key=os.getenv("ARGILLA_API_KEY")
        )

        rg_dataset = dataset.to_argilla()
        rg_dataset.push_to_argilla(name=f"my-dataset-{uuid4()}", workspace="admin")
    except ImportError:
        pass
