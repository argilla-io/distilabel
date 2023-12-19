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
from distilabel.llm import LLMPool, ProcessLLM, vLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask, UltraFeedbackTask
from vllm import LLM


def load_notus(task):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="TheBloke/notus-7B-v1-AWQ", quantization="awq")
    return vLLM(vllm=llm, task=task, max_new_tokens=512, prompt_format="notus")


def load_zephyr(task):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(model="TheBloke/zephyr-7b-beta-AWQ", quantization="awq")
    return vLLM(vllm=llm, task=task, max_new_tokens=512, prompt_format="zephyr")


def load_openai(task):
    from distilabel.llm import OpenAILLM

    return OpenAILLM(
        model="gpt-3.5-turbo",
        task=task,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_new_tokens=512,
    )


if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )

    pipeline = Pipeline(
        generator=LLMPool(
            [
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_notus),
                ProcessLLM(task=TextGenerationTask(), load_llm_fn=load_zephyr),
            ]
        ),
        labeller=ProcessLLM(
            task=UltraFeedbackTask.for_instruction_following(), load_llm_fn=load_openai
        ),
    )

    dataset = pipeline.generate(
        dataset=dataset,
        num_generations=3,
        batch_size=5,
    )
