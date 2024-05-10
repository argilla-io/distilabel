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

from distilabel.llms import LlamaCppLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.generators.data import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

if __name__ == "__main__":
    start_time = time.time()

    with Pipeline(name="ultrafeedback-dpo") as pipeline:
        # load_dataset = LoadHubDataset(
        #     name="load_dataset",
        #     output_mappings={"prompt": "instruction"},
        # )
        load_dataset = LoadDataFromDicts(
            name="load_dataset",
            data=[
                {"instruction": "Tell me a joke."},
            ],
        )
        from pathlib import Path

        # model_path = str(Path.home() / "Downloads/zephyr-7b-beta.Q4_K_M.gguf")
        model_path = str(
            Path.home() / "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"
        )

        text_generation_zephyr = TextGeneration(
            name="text_generation_zephyr",
            llm=LlamaCppLLM(
                model_path=model_path,  # type: ignore
                n_gpu_layers=-1,
            ),
            # llm=MistralLLM(model="mistral-tiny", api_key=os.getenv("MISTRAL_API_KEY")),
            input_batch_size=10,
            output_mappings={"model_name": "generation_model"},
        )
        load_dataset.connect(text_generation_zephyr)

    distiset = pipeline.run(
        parameters={
            # "load_dataset": {
            #     "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
            #     "split": "test",
            # },
            "wrong_step": {"runtime": "value"},
            "text_generation_gemma": {
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
    print(distiset)
    distiset.push_to_hub(
        "distilabel-internal-testing/test-dockerfile",
        token="hf",
    )
