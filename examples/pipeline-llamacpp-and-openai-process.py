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

# WARNING: to run this example in Mac OS use:
# no_proxy=* OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python examples/pipeline-llamacpp-and-openai-process.py
# Otherwise you will get an error when loading the llama.cpp model

import os
from typing import TYPE_CHECKING

from datasets import load_dataset
from distilabel.llm import ProcessLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from llama_cpp import Llama

if TYPE_CHECKING:
    from distilabel.llm import LLM
    from distilabel.tasks import Task


def load_llama_cpp_llm(task: "Task") -> "LLM":
    from distilabel.llm import LlamaCppLLM

    llama = Llama(
        model_path="<PATH_TO_GGUF_MODEL>", n_gpu_layers=10, n_ctx=1024, verbose=False
    )
    return LlamaCppLLM(
        model=llama, task=task, max_new_tokens=512, prompt_format="zephyr"
    )


def load_openai_llm(task: "Task") -> "LLM":
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-3.5-turbo",
        task=task,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        num_threads=2,
        max_new_tokens=512,
    )


if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )

    pipeline = Pipeline(
        generator=ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_llama_cpp_llm),
        labeller=ProcessLLM(
            task=UltraFeedbackTask.for_text_quality(), load_llm_fn=load_openai_llm
        ),
    )

    dataset = pipeline.generate(
        dataset,  # type: ignore
        num_generations=2,
        batch_size=1,
        enable_checkpoints=True,
        display_progress_bar=False,
    )

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
