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

import argilla as rg
import pytest
from distilabel.dataset import CustomDataset
from distilabel.tasks.text_generation.self_instruct import (
    SelfInstructTask,
)

breadth = """I want you to act as a Prompt Creator.
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
This new prompt should belong to the same domain as the #Given Prompt# but must be even more rare.
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.
The #Created Prompt# must be reasonable and must be understood and responded by humans.
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#
#Given Prompt#:
HELLO

#Created Prompt#:
"""

default_prompt = """# Task Description
Develop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model's textual capabilities.

# Criteria for Queries
Incorporate a diverse range of verbs, avoiding repetition.
Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.
Design queries to be self-contained and standalone.
Blend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.
Write each query on a separate line and avoid using numbered lists or bullet points.

# AI Application
AI assistant

# Context
HELLO

# Output
"""

custom_prompt = """# Task Description
Develop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model's textual capabilities.

# Criteria for Queries
Incorporate a diverse range of verbs, avoiding repetition.
Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.
Design queries to be self-contained and standalone.
Write each query on a separate line and avoid using numbered lists or bullet points.

# AI Application
An AI assistant adept at writing Haiku.
It expects complete suggestions from users providing details of the kind of haiku they want.
The AI assistant will help users write haiku about particular topics and is willing to accept requests related to a specific subject or object or a more abstract request
based on an emotion, theme or vibe.

# Context
HELLO

# Output
"""

application_description = """An AI assistant adept at writing Haiku.
It expects complete suggestions from users providing details of the kind of haiku they want.
The AI assistant will help users write haiku about particular topics and is willing to accept requests related to a specific subject or object or a more abstract request
based on an emotion, theme or vibe."""

criteria_for_query_generation = """Incorporate a diverse range of verbs, avoiding repetition.
Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.
Design queries to be self-contained and standalone."""


@pytest.mark.parametrize(
    "self_instruct, expected",
    [
        (SelfInstructTask(), default_prompt),
        (
            SelfInstructTask(
                application_description=application_description,
                criteria_for_query_generation=criteria_for_query_generation,
            ),
            custom_prompt,
        ),
    ],
)
def test_self_instruct_task(self_instruct: SelfInstructTask, expected: str):
    assert isinstance(self_instruct, SelfInstructTask)
    assert (
        self_instruct.system_prompt
        == "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application."
    )
    assert self_instruct.generate_prompt(input="HELLO").formatted_prompt == expected


@pytest.fixture
def custom_self_instruct_dataset() -> CustomDataset:
    ds = CustomDataset.from_dict(
        {
            "input": [
                "EN EN\nEUROPEAN\nCOMMISSION\nProposal for a\nREGULATION OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL\nLAYING DOWN HARMONISED RULES ON ARTIFICIAL INTELLIGENCE\n(ARTIFICIAL INTELLIGENCE ACT) AND AMENDING CERTAIN UNION\nLEGISLATIVE ACTS\x0cEN\nEXPLANATORY MEMORANDUM\n1. CONTEXT OF THE PROPOSAL\n1.1. Reasons for and objectives of the proposal\nThis explanatory memorandum accompanies the proposal for a Regulation laying down\nharmonised rules on artificial intelligence (Artificial Intelligence Act). Artificial Intelligence\n(AI) is a fast evolving family of technologies that can bring a wide array of economic and\nsocietal benefits across the entire spectrum of industries and social activities. By improving\nprediction, optimising operations and resource allocation, and personalising service delivery,\nthe use of artificial intelligence can support socially and environmentally beneficial outcomes\nand provide key competitive advantages to companies and the European economy. ",
                "EN EN\nEUROPEAN\nCOMMISSION\nProposal for a\nREGULATION OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL\nLAYING DOWN HARMONISED RULES ON ARTIFICIAL INTELLIGENCE\n(ARTIFICIAL INTELLIGENCE ACT) AND AMENDING CERTAIN UNION\nLEGISLATIVE ACTS\x0cEN\nEXPLANATORY MEMORANDUM\n1. CONTEXT OF THE PROPOSAL\n1.1. Reasons for and objectives of the proposal\nThis explanatory memorandum accompanies the proposal for a Regulation laying down\nharmonised rules on artificial intelligence (Artificial Intelligence Act). Artificial Intelligence\n(AI) is a fast evolving family of technologies that can bring a wide array of economic and\nsocietal benefits across the entire spectrum of industries and social activities. By improving\nprediction, optimising operations and resource allocation, and personalising service delivery,\nthe use of artificial intelligence can support socially and environmentally beneficial outcomes\nand provide key competitive advantages to companies and the European economy. ",
            ],
            "generation_model": [["argilla/notus-7b-v1"], ["argilla/notus-7b-v1"]],
            "generation_prompt": [
                [
                    'You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.\n# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nA assistant that can answer questions about the AI Act made by the European Union.\n\n# Context\nEN EN\nEUROPEAN\nCOMMISSION\nProposal for a\nREGULATION OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL\nLAYING DOWN HARMONISED RULES ON ARTIFICIAL INTELLIGENCE\n(ARTIFICIAL INTELLIGENCE ACT) AND AMENDING CERTAIN UNION\nLEGISLATIVE ACTS\x0cEN\nEXPLANATORY MEMORANDUM\n1. CONTEXT OF THE PROPOSAL\n1.1. Reasons for and objectives of the proposal\nThis explanatory memorandum accompanies the proposal for a Regulation laying down\nharmonised rules on artificial intelligence (Artificial Intelligence Act). Artificial Intelligence\n(AI) is a fast evolving family of technologies that can bring a wide array of economic and\nsocietal benefits across the entire spectrum of industries and social activities. By improving\nprediction, optimising operations and resource allocation, and personalising service delivery,\nthe use of artificial intelligence can support socially and environmentally beneficial outcomes\nand provide key competitive advantages to companies and the European economy. \n\n# Output\n'
                ],
                [
                    'You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application.\n# Task Description\nDevelop 5 user queries that can be received by the given AI application and applicable to the provided context. Emphasize diversity in verbs and linguistic structures within the model\'s textual capabilities.\n\n# Criteria for Queries\nIncorporate a diverse range of verbs, avoiding repetition.\nEnsure queries are compatible with AI model\'s text generation functions and are limited to 1-2 sentences.\nDesign queries to be self-contained and standalone.\nBlend interrogative (e.g., "What is the significance of x?") and imperative (e.g., "Detail the process of x.") styles.\nWrite each query on a separate line and avoid using numbered lists or bullet points.\n\n# AI Application\nA assistant that can answer questions about the AI Act made by the European Union.\n\n# Context\nEN EN\nEUROPEAN\nCOMMISSION\nProposal for a\nREGULATION OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL\nLAYING DOWN HARMONISED RULES ON ARTIFICIAL INTELLIGENCE\n(ARTIFICIAL INTELLIGENCE ACT) AND AMENDING CERTAIN UNION\nLEGISLATIVE ACTS\x0cEN\nEXPLANATORY MEMORANDUM\n1. CONTEXT OF THE PROPOSAL\n1.1. Reasons for and objectives of the proposal\nThis explanatory memorandum accompanies the proposal for a Regulation laying down\nharmonised rules on artificial intelligence (Artificial Intelligence Act). Artificial Intelligence\n(AI) is a fast evolving family of technologies that can bring a wide array of economic and\nsocietal benefits across the entire spectrum of industries and social activities. By improving\nprediction, optimising operations and resource allocation, and personalising service delivery,\nthe use of artificial intelligence can support socially and environmentally beneficial outcomes\nand provide key competitive advantages to companies and the European economy. \n\n# Output\n'
                ],
            ],
            "raw_generation_responses": [
                [
                    "1. What are the reasons for and objectives of the proposal for a Regulation laying down harmonised rules on artificial intelligence?\n2. How can artificial intelligence improve prediction, optimise operations and resource allocation, and personalise service delivery?\n3. What benefits can artificial intelligence bring to the European economy and society as a whole?\n4. How can the use of artificial intelligence support socially and environmentally beneficial outcomes?\n5. What competitive advantages can companies gain from using artificial intelligence?"
                ],
                [
                    "1. What are the reasons for and objectives of the proposal for a Regulation laying down harmonised rules on artificial intelligence?\n2. How can artificial intelligence improve prediction, optimise operations and resource allocation, and personalise service delivery?\n3. What benefits can artificial intelligence bring to the European economy and society as a whole?\n4. How can the use of artificial intelligence support socially and environmentally beneficial outcomes?\n5. What competitive advantages can companies gain from using artificial intelligence?"
                ],
            ],
            "instructions": [
                [
                    "What are the reasons for and objectives of the proposal for a Regulation laying down harmonised rules on artificial intelligence?",
                    "How can artificial intelligence improve prediction, optimise operations and resource allocation, and personalise service delivery?",
                    "What benefits can artificial intelligence bring to the European economy and society as a whole?",
                    "How can the use of artificial intelligence support socially and environmentally beneficial outcomes?",
                ],
                [
                    "What are the reasons for and objectives of the proposal for a Regulation laying down harmonised rules on artificial intelligence?",
                    "How can artificial intelligence improve prediction, optimise operations and resource allocation, and personalise service delivery?",
                    "What benefits can artificial intelligence bring to the European economy and society as a whole?",
                    "How can the use of artificial intelligence support socially and environmentally beneficial outcomes?",
                ],
            ],
        }
    )
    ds.task = SelfInstructTask()
    return ds


def test_self_instruct_task_to_argilla_dataset(custom_self_instruct_dataset):
    ds_row = custom_self_instruct_dataset[0]
    task = custom_self_instruct_dataset.task
    rg_dataset = task.to_argilla_dataset(ds_row)
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 0


def test_self_instruct_task_to_argilla_record(custom_self_instruct_dataset):
    ds_row = custom_self_instruct_dataset[0]
    task = custom_self_instruct_dataset.task
    records = task.to_argilla_record(ds_row)
    assert isinstance(records, list)
    assert len(records) == 4
    assert isinstance(records[0], rg.FeedbackRecord)


def test_self_instruct_task_to_argilla(custom_self_instruct_dataset):
    rg_dataset = custom_self_instruct_dataset.to_argilla(
        vector_strategy=False, metric_strategy=False
    )
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 8
