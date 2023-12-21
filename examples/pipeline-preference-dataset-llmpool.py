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

from datasets import load_dataset
from distilabel.llm import LLM, LLMPool, ProcessLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import Task, TextGenerationTask, UltraFeedbackTask


def load_notus(task: Task) -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    return vLLM(
        vllm=LLM(model="argilla/notus-7b-v1"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_zephyr(task: Task) -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    return vLLM(
        vllm=LLM(model="HuggingFaceH4/zephyr-7b-beta"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_starling(task: Task) -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    return vLLM(
        vllm=LLM(model="berkeley-nest/Starling-LM-7B-alpha"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_neural_chat(task: Task) -> LLM:
    import os

    from distilabel.llm import vLLM
    from vllm import LLM

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    return vLLM(
        vllm=LLM(model="Intel/neural-chat-7b-v3-3"),
        task=task,
        max_new_tokens=512,
        temperature=0.7,
        prompt_format="notus",
    )


def load_gpt_4(task: UltraFeedbackTask) -> LLM:
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-4-1106-preview",
        task=task,
        max_new_tokens=512,
        num_threads=4,
    )


if __name__ == "__main__":
    pipeline = Pipeline(
        generator=LLMPool(
            llms=[
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_notus),
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_zephyr),
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_starling),
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_neural_chat),
            ]
        ),
        labeller=ProcessLLM(
            task=UltraFeedbackTask.for_instruction_following(), load_llm_fn=load_gpt_4
        ),
    )

    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[:50]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )

    dataset = pipeline.generate(
        dataset=dataset, num_generations=2, batch_size=10, display_progress_bar=True
    )

    rg_argilla = dataset.to_argilla()
    rg_argilla.push_to_argilla(name="preference-dataset", workspace="admin")
