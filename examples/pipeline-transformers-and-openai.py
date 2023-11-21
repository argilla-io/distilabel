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

import torch
from datasets import load_dataset
from distilabel.llm import OpenAILLM, TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )

    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", dtype=torch.bfloat16, device="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.padding_side = "left"

    pipeline = Pipeline(
        generator=TransformersLLM(
            model=model,
            tokenizer=tokenizer,
            task=TextGenerationTask(),
            max_new_tokens=128,
            temperature=0.3,
            prompt_format="zephyr",
        ),
        labeller=OpenAILLM(
            model="gpt-3.5-turbo",
            task=UltraFeedbackTask.for_instruction_following(),
            max_new_tokens=128,
            num_threads=2,
            openai_api_key="<OPENAI_API_KEY>",
            temperature=0.0,
        ),
    )

    dataset = pipeline.generate(
        dataset,  # type: ignore
        num_generations=2,
        batch_size=1,
        enable_checkpoints=True,
        display_progress_bar=True,
    )

    # Push to the HuggingFace Hub
    dataset.push_to_hub(
        os.getenv("HF_REPO_ID"),  # type: ignore
        split="train",
        private=True,
    )

    try:
        from uuid import uuid4

        import argilla as rg

        rg.init(
            api_url="<ARGILLA_API_URL>",
            api_key="<ARGILLA_API_KEY>",
        )

        # Convert into an Argilla dataset and push it to Argilla
        rg_dataset = dataset.to_argilla()
        rg_dataset.push_to_argilla(
            name=f"my-dataset-{uuid4()}",
            workspace="<ARGILLA_WORKSPACE_NAME>",
        )
    except ImportError:
        pass
