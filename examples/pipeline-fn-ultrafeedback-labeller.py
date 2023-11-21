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
from distilabel.pipeline import pipeline

if __name__ == "__main__":
    dataset = (
        load_dataset("ProlificAI/social-reasoning-rlhf", split="train[:10]")
        .rename_column("question", "input")
        .map(lambda x: {"generations": [x["chosen"], x["rejected"]]})
        .remove_columns(["chosen", "rejected"])
    )

    pipe = pipeline(
        "preference",
        "honesty",
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
        display_progress_bar=True,
        enable_checkpoints=True,
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
