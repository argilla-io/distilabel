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

from distilabel.llm.openai import OpenAILLM
from distilabel.pipeline.local import Pipeline
from distilabel.steps.generators.huggingface import LoadHubDataset
from distilabel.steps.globals.huggingface import PushToHub
from distilabel.steps.task.self_instruct import SelfInstruct

if __name__ == "__main__":
    start_time = time.time()

    with Pipeline() as pipeline:
        load_hub_dataset = LoadHubDataset(
            name="load_hub_dataset", output_mappings={"prompt": "input"}
        )
        self_instruct = SelfInstruct(
            name="self_instruct",
            llm=OpenAILLM(
                model="gpt-4",
                api_key="",  # type: ignore
            ),
        )

        push_to_hub = PushToHub(name="push_to_hub")  # type: ignore

        load_hub_dataset.connect(self_instruct)
        self_instruct.connect(push_to_hub)

    pipeline.run(
        parameters={
            "load_hub_dataset": {
                "repo_id": "HuggingFaceH4/instruction-dataset",
                "split": "test",
            },
            "push_to_hub": {
                "repo_id": "ignacioct/selfinstruct",
                "split": "train",
                "private": False,
                "token": "",
            },
        }
    )

    print("--- %s seconds ---" % (time.time() - start_time))
