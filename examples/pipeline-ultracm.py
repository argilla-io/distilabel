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

from datasets import Dataset
from distilabel.llm import LlamaCppLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks.critique.ultracm import UltraCMTask
from llama_cpp import Llama

if __name__ == "__main__":
    pipeline = Pipeline(
        labeller=LlamaCppLLM(
            model=Llama(model_path="./UltraCM-13b.q4_k_m.gguf"),
            task=UltraCMTask(),
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=1024,
            repeat_penalty=1.2,
            seed=-1,  # is the same as do_sample=True,
        ),
    )

    dataset = Dataset.from_dict(
        {
            "instruction": ["What's the capital of Spain?"],
            "completion": ["Madrid"],
        }
    )

    dataset = pipeline.generate(
        dataset,  # type: ignore
        display_progress_bar=True,
    )
