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
from distilabel.llm import LlamaCppLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Llama2TextGenerationTask, UltraFeedbackTask
from llama_cpp import Llama

if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )

    pipeline = Pipeline(
        generator=LlamaCppLLM(
            model=Llama(
                model_path="<PATH_TO_GGUF_MODEL>", n_gpu_layers=-1
            ),  # e.g. https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_0.gguf
            task=Llama2TextGenerationTask(),
            max_new_tokens=128,
            temperature=0.3,
        ),
        labeller=OpenAILLM(
            model="gpt-3.5-turbo",
            task=UltraFeedbackTask.for_text_quality(),
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
