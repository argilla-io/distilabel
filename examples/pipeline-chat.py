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
from uuid import uuid4

import argilla as rg
from datasets import Dataset
from distilabel.llm.openai import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks.chat import ChatTask

if __name__ == "__main__":
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that always answers like a pirate.",
                    },
                    {
                        "role": "user",
                        "content": "What's the most terrifying creature in the sea?",
                    },
                ]
            ],
        }
    )

    generator = OpenAILLM(
        model="gpt-4",
        task=ChatTask(input_format="openai", output_format="openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY", None),
        max_new_tokens=1024,
        prompt_format=None,
    )

    pipeline = Pipeline(generator=generator)
    dataset = pipeline.generate(
        dataset=dataset,
        shuffle_before_labelling=False,
        num_generations=2,
        skip_dry_run=True,
        display_progress_bar=True,
    )  # type: ignore

    rg_dataset = dataset.to_argilla()

    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY"),
    )

    rg_dataset.push_to_argilla(name=f"chat-{uuid4()}", workspace="admin")
