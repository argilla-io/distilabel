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

from typing import List

from pydantic import BaseModel, Field

from distilabel.llms import MistralLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


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


with Pipeline(
    name="Knowledge-Graphs",
    description=(
        "Generate knowledge graphs to answer questions, this type of dataset can be used to "
        "steer a model to answer questions with a knowledge graph."
    ),
) as pipeline:
    sample_questions = [
        "Teach me about quantum mechanics",
        "Who is who in The Simpsons family?",
        "Tell me about the evolution of programming languages",
    ]

    load_dataset = LoadDataFromDicts(
        name="load_instructions",
        data=[
            {
                "system_prompt": "You are a knowledge graph expert generator. Help me understand by describing everything as a detailed knowledge graph.",
                "instruction": f"{question}",
            }
            for question in sample_questions
        ],
    )

    text_generation = TextGeneration(
        name="knowledge_graph_generation",
        llm=MistralLLM(
            model="open-mixtral-8x22b", structured_output={"schema": KnowledgeGraph}
        ),
        input_batch_size=8,
        output_mappings={"model_name": "generation_model"},
    )
    load_dataset >> text_generation


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {"generation_kwargs": {"max_new_tokens": 2048}}
            }
        },
        use_cache=False,
    )

    distiset.push_to_hub("distilabel-internal-testing/knowledge_graphs")
