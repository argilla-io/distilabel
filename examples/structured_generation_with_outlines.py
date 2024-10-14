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

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, StringConstraints, conint
from typing_extensions import Annotated

from distilabel.llms import LlamaCppLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


class Weapon(str, Enum):
    sword = "sword"
    axe = "axe"
    mace = "mace"
    spear = "spear"
    bow = "bow"
    crossbow = "crossbow"


class Armor(str, Enum):
    leather = "leather"
    chainmail = "chainmail"
    plate = "plate"
    mithril = "mithril"


class Character(BaseModel):
    name: Annotated[str, StringConstraints(max_length=30)]
    age: conint(gt=1, lt=3000)
    armor: Armor
    weapon: Weapon


# Download the model with
# curl -L -o ~/Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf

model_path = "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

with Pipeline("RPG-characters") as pipeline:
    system_prompt = (
        "You are a leading role play gamer. You have seen thousands of different characters and their attributes."
        " Please return a JSON object with common attributes of an RPG character."
    )

    load_dataset = LoadDataFromDicts(
        name="load_instructions",
        data=[
            {
                "system_prompt": system_prompt,
                "instruction": f"Give me a character description for a {char}",
            }
            for char in ["dwarf", "elf", "human", "ork"]
        ],
    )
    llm = LlamaCppLLM(
        model_path=str(Path.home() / model_path),  # type: ignore
        n_gpu_layers=-1,
        n_ctx=1024,
        structured_output={"format": "json", "schema": Character},
    )
    # Change to vLLM as such:
    # llm = vLLM(
    #     model="teknium/OpenHermes-2.5-Mistral-7B",
    #     extra_kwargs={"tensor_parallel_size": 1},
    #     structured_output={"format": "json", "schema": Character},
    # )

    text_generation = TextGeneration(
        name="text_generation_rpg",
        llm=llm,
        input_batch_size=8,
        output_mappings={"model_name": "generation_model"},
    )
    load_dataset >> text_generation


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {"generation_kwargs": {"max_new_tokens": 256}}
            }
        },
        use_cache=False,
    )
    for num, character in enumerate(distiset["default"]["train"]["generation"]):
        print(f"Character: {num}")
        print(character)

# Character: 0
# {
# "name": "Gimli",
# "age": 42,
# "armor": "plate",
# "weapon": "axe" }
# Character: 1
# {"name":"Gaelen","age":600,"armor":"leather","weapon":"bow"}
# Character: 2
# {"name": "John Smith","age": 35,"armor": "leather","weapon": "sword"}
# Character: 3
# { "name": "Grug", "age": 35, "armor": "leather", "weapon": "axe"}
