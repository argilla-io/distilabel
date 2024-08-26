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
from typing import List

from pydantic import BaseModel, Field, StringConstraints, conint
from typing_extensions import Annotated

from distilabel.steps.tasks.structured_outputs.utils import json_schema_to_model


class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)


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


def test_json_schema_to_model():
    assert type(json_schema_to_model(Node.model_json_schema())) is type(Node)


def test_json_schema_to_model_with_enum():
    assert type(json_schema_to_model(Character.model_json_schema())) is type(Character)


def test_json_schema_to_model_nested():
    assert type(json_schema_to_model(KnowledgeGraph.model_json_schema())) is type(
        KnowledgeGraph
    )
