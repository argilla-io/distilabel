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

from datasets import Dataset
from distilabel.llm.replicate import ReplicateLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask

if __name__ == "__main__":
    dataset = Dataset.from_dict(
        {
            "input": ["What are the tallest buildings on earth?"],
        }
    )

    generator = ReplicateLLM(
        endpoint_name="titocosta/notus-7b-v1",
        endpoint_revision="dbcd2277b32873525e618545e13e64c3ba121b681cbd2b5f0ee7f95325e7a395",
        replicate_api_token=os.getenv("REPLICATE_API_TOKEN", None),
        task=TextGenerationTask(),
        generation_kwargs={
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.2,
            "max_new_tokens": 512,
            "system_message": "You are a helpful AI assistant",
            "prompt_template": "<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n",
        },
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

    print(dataset[0])
