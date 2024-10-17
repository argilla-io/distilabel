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

from pathlib import Path

from datasets import load_dataset

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import CombineOutputs, DataSampler, LoadDataFromDicts
from distilabel.steps.tasks import (
    APIGenExecutionChecker,
    APIGenGenerator,
    APIGenSemanticChecker,
)
from distilabel.steps.tasks.apigen.utils import PrepareExamples, load_module_from_path

libpath = Path(__file__).parent / "lib_apigen.py"

data = [
    {
        "func_name": "final_velocity",
        "func_desc": "Calculates the final velocity of an object given its initial velocity, acceleration, and time.",
    },
    {
        "func_name": "permutation_count",
        "func_desc": "Calculates the number of permutations of k elements from a set of n elements.",
    },
    {
        "func_name": "getdivision",
        "func_desc": "Divides two numbers by making an API call to a division service.",
    },
    {
        "func_name": "binary_addition",
        "func_desc": "Adds two binary numbers and returns the result as a binary string.",
    },
    {
        "func_name": "swapi_planet_resource",
        "func_desc": "get a specific planets resource",
    },
    {
        "func_name": "disney_character",
        "func_desc": "Find a specific character using this endpoint",
    },
]

libpath_module = load_module_from_path(libpath)
tools = libpath_module.get_tools()  # call get_tools()

# TODO: Add in the tools between 0 and 2 extra tools to make the task more challenging.
for row in data:
    # The tools should have a mix where both the correct and irrelevant tools are present.
    row.update({"tools": [tools[row["func_name"]]]})


ds_og = (
    load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    .shuffle(seed=42)
    .select(range(500))
    .to_list()
)


with Pipeline(name="APIGenPipeline") as pipeline:
    loader_seeds = LoadDataFromDicts(data=data)
    sampler = DataSampler(
        data=ds_og,
        size=2,
        samples=len(data),
        batch_size=8,
    )

    prep_examples = PrepareExamples()

    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    llm = InferenceEndpointsLLM(
        model_id=model_id,
        tokenizer_id=model_id,
        generation_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 2048,
        },
    )
    apigen = APIGenGenerator(
        llm=llm,
        use_default_structured_output=True,
    )
    combine_steps = CombineOutputs()

    execution_checker = APIGenExecutionChecker(libpath=str(libpath))
    semantic_checker = APIGenSemanticChecker(llm=llm)

    sampler >> prep_examples
    (
        [loader_seeds, prep_examples]
        >> combine_steps
        >> apigen
        >> execution_checker
        >> semantic_checker
    )


if __name__ == "__main__":
    distiset = pipeline.run()
    print(distiset["default"]["train"][0])
