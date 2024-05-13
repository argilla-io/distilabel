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

import time
from pathlib import Path

from distilabel.llms import LlamaCppLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

if __name__ == "__main__":
    start_time = time.time()

    with Pipeline(name="test-pipe") as pipeline:
        load_dataset = LoadDataFromDicts(
            data=[
                {"instruction": "Tell me a joke."},
            ],
            name="load_dataset",
        )
        model_path = str(
            Path.home() / "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
        )
        text_generation = TextGeneration(
            llm=LlamaCppLLM(
                model_path=model_path,  # type: ignore
                n_gpu_layers=-1,
                n_ctx=1024,
            ),
            input_batch_size=10,
            output_mappings={"model_name": "generation_model"},
            name="text_generation",
        )
        load_dataset.connect(text_generation)

    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 1024,
                        "temperature": 0.7,
                    },
                },
            },
        },
        use_cache=False,
    )
    print("--- %s seconds ---" % (time.time() - start_time))
    distiset.push_to_hub("distilabel-internal-testing/test-dockerfile")
