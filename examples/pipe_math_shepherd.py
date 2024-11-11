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

from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import CombineOutputs
from distilabel.steps.tasks.math_shepherd.completer import MathShepherdCompleter
from distilabel.steps.tasks.math_shepherd.generator import MathShepherdGenerator

ds_name = "openai/gsm8k"

ds = (
    load_dataset(ds_name, "main", split="test")
    .rename_column("question", "instruction")
    .select(range(3))
)


with Pipeline(name="Math-Shepherd") as pipe:
    model_id_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    model_id_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    llm_70B = InferenceEndpointsLLM(
        model_id=model_id_8B,
        tokenizer_id=model_id_8B,
        generation_kwargs={"max_new_tokens": 1024, "temperature": 0.6},
    )
    llm_8B = InferenceEndpointsLLM(
        model_id=model_id_8B,
        tokenizer_id=model_id_8B,
        generation_kwargs={"max_new_tokens": 2048, "temperature": 0.6},
    )

    generator_golden = MathShepherdGenerator(
        name="golden_generator",
        llm=llm_70B,
    )
    generator = MathShepherdGenerator(
        name="generator",
        llm=llm_8B,
        M=5,  # Generate 6 sample solutions
    )
    completer = MathShepherdCompleter(name="completer", llm=llm_8B, N=4)

    combine = CombineOutputs()
    [generator_golden, generator] >> combine >> completer


if __name__ == "__main__":
    distiset = pipe.run(use_cache=False, dataset=ds)
    distiset.push_to_hub("plaguss/test_math_shepherd_v4")
