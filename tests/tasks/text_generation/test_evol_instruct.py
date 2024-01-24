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
from distilabel.tasks.text_generation.evol_instruct import (
    EvolInstructTask,
    _get_stopwords,
)


def test_get_stopwords():
    stopwords = _get_stopwords()
    assert len(stopwords) == 179


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

constraints = """I want you to act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
Please add one more constraints/requirements into #The Given Prompt#
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#:
HELLO

#Rewritten Prompt#:
"""

deepen = """I want you to act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#:
HELLO

#Rewritten Prompt#:
"""

concretizing = """I want you to act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
Please replace general concepts with more specific concepts.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#:
HELLO

#Rewritten Prompt#:
"""

reasoning = """I want you to act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#:
HELLO

#Rewritten Prompt#:
"""


@pytest.mark.parametrize(
    "evolution_method, expected",
    [
        ("breadth", breadth),
        ("constraints", constraints),
        ("deepen", deepen),
        ("concretizing", concretizing),
        ("reasoning", reasoning),
    ],
)
def test_evol_instruct_task(evolution_method: str, expected: str):
    task = EvolInstructTask()
    assert isinstance(task, EvolInstructTask)
    assert task.system_prompt == ""
    assert (
        task.generate_prompt(
            input="HELLO", evolution_method=evolution_method
        ).formatted_prompt
        == expected
    )


@pytest.fixture
def custom_evol_instruct_dataset() -> CustomDataset:
    ds = CustomDataset.from_dict(
        {
            "input": [
                'Create a sentence using the words "happy," "joyful," and "thrilled."\n',
                "Construct plumbing diagrams for a two-story house\n",
            ],
            "generation_model": [["gpt-3.5-turbo"], ["gpt-3.5-turbo"]],
            "generation_prompt": [
                [
                    [
                        {"content": "", "role": "system"},
                        {
                            "content": "I want you to act as a Prompt Rewriter.\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\nYou SHOULD complicate the given prompt using the following method:\nIf #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n#The Given Prompt#:\nCreate a sentence using the words \"happy,\" \"joyful,\" and \"thrilled.\"\n\n\n#Rewritten Prompt#:\n",
                            "role": "user",
                        },
                    ]
                ],
                [
                    [
                        {"content": "", "role": "system"},
                        {
                            "content": "I want you to act as a Prompt Rewriter.\nYour objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\nBut the rewritten prompt must be reasonable and must be understood and responded by humans.\nYour rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\nYou SHOULD complicate the given prompt using the following method:\nIf #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.\nYou should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n#The Given Prompt#:\nConstruct plumbing diagrams for a two-story house\n\n\n#Rewritten Prompt#:\n",
                            "role": "user",
                        },
                    ]
                ],
            ],
            "raw_generation_responses": [
                [
                    'Compose a concise and articulate sentence that incorporates the terms "ecstatic," "exhilarated," "blissful," and "overjoyed."'
                ],
                [
                    "Design comprehensive and detailed plumbing diagrams for a two-story house that include separate diagrams for each floor, showcasing the layout, dimensions, and connections of all plumbing fixtures, pipelines, drains, vents, water supply sources, and sewage disposal systems. These diagrams should consider the specific requirements and codes of local building regulations while providing a clear and accurate representation of the plumbing infrastructure throughout the entire house."
                ],
            ],
            "instruction": [
                [
                    'Compose a concise and articulate sentence that incorporates the terms "ecstatic," "exhilarated," "blissful," and "overjoyed."'
                ],
                [
                    "Design comprehensive and detailed plumbing diagrams for a two-story house that include separate diagrams for each floor, showcasing the layout, dimensions, and connections of all plumbing fixtures, pipelines, drains, vents, water supply sources, and sewage disposal systems. These diagrams should consider the specific requirements and codes of local building regulations while providing a clear and accurate representation of the plumbing infrastructure throughout the entire house."
                ],
            ],
        }
    )
    ds.task = EvolInstructTask()
    return ds


def test_evol_instruct_task_to_argilla_dataset(custom_evol_instruct_dataset):
    ds_row = custom_evol_instruct_dataset[0]
    task = custom_evol_instruct_dataset.task
    rg_dataset = task.to_argilla_dataset(ds_row)
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 0


def test_evol_instruct_task_to_argilla_record(custom_evol_instruct_dataset):
    ds_row = custom_evol_instruct_dataset[0]
    task = custom_evol_instruct_dataset.task
    records = task.to_argilla_record(ds_row)
    assert isinstance(records, list)
    assert len(records) == 1
    assert isinstance(records[0], rg.FeedbackRecord)


def test_evol_instruct_task_to_argilla(custom_evol_instruct_dataset):
    rg_dataset = custom_evol_instruct_dataset.to_argilla(
        vector_strategy=False, metric_strategy=False
    )
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 2
