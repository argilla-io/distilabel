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
from distilabel.tasks.text_generation.evol_complexity import EvolComplexityTask
from distilabel.tasks.text_generation.evol_instruct import (
    EvolInstructTask,
    _get_stopwords,
)
from distilabel.tasks.text_generation.evol_quality import EvolQualityTask


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

base_evol_instruct = """I want you to act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
"""

end_evol_instruct = """
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
#The Given Prompt#:
HELLO
#Rewritten Prompt#:
"""

constraints = f"{base_evol_instruct}Please add one more constraints/requirements into #The Given Prompt#{end_evol_instruct}"
deepen = f"{base_evol_instruct}If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.{end_evol_instruct}"
concretizing = f"{base_evol_instruct}Please replace general concepts with more specific concepts.{end_evol_instruct}"
reasoning = f"{base_evol_instruct}If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.{end_evol_instruct}"

base_evol_quality = """I want you to act as a Response Rewriter
Your goal is to enhance the quality of the response given by an AI assistant
to the #Given Prompt# through rewriting.
But the rewritten response must be reasonable and must be understood by humans.
Your rewriting cannot omit the non-text parts such as the table and code in
#Given Prompt# and #Given Response#. Also, please do not omit the input
in #Given Prompt#.
You Should enhance the quality of the response using the following method:
"""

end_evol_quality = """
You should try your best not to make the #Rewritten Response# become verbose,
#Rewritten Response# can only add 10 to 20 words into #Given Response#.
'#Given Response#', '#Rewritten Response#', 'given response' and 'rewritten response'
are not allowed to appear in #Rewritten Response#
#Given Prompt#:
HELLO
#Given Response#:
HELLO
#Rewritten Response#:"""

helpfulness = f"{base_evol_quality}Please make the Response more helpful to the user.{end_evol_quality}"
relevance = f"{base_evol_quality}Please make the Response more relevant to #Given Prompt#.{end_evol_quality}"
depth = f"{base_evol_quality}Please make the Response more in-depth.{end_evol_quality}"
creativity = f"{base_evol_quality}Please increase the creativity of the response.{end_evol_quality}"
details = f"{base_evol_quality}Please increase the detail level of Response.{end_evol_quality}"


@pytest.mark.parametrize(
    "evolution_method, expected, instruct_type",
    [
        ("breadth", breadth, EvolInstructTask),
        ("constraints", constraints, EvolInstructTask),
        ("deepen", deepen, EvolInstructTask),
        ("concretizing", concretizing, EvolInstructTask),
        ("reasoning", reasoning, EvolInstructTask),
        ("constraints", constraints, EvolComplexityTask),
        ("deepen", deepen, EvolComplexityTask),
        ("concretizing", concretizing, EvolComplexityTask),
        ("reasoning", reasoning, EvolComplexityTask),
        ("helpfulness", helpfulness, EvolQualityTask),
        ("relevance", relevance, EvolQualityTask),
        ("depth", depth, EvolQualityTask),
        ("creativity", creativity, EvolQualityTask),
        ("details", details, EvolQualityTask),
        ("fake", ValueError, EvolInstructTask),
        ("fake", ValueError, EvolQualityTask),
        # # breadth is not a valid evolution method for EvolComplexityTask
        ("breadth", ValueError, EvolComplexityTask),
    ],
)
def test_evol_task(evolution_method: str, expected: str, instruct_type: object):
    mock_kwargs = {"input": "HELLO", "generation": "HELLO"}
    task = instruct_type()
    assert isinstance(task, instruct_type)
    assert task.system_prompt == ""
    if isinstance(expected, str):
        print("---COSA")
        print(
            task.generate_prompt(
                **mock_kwargs, evolution_method=evolution_method
            ).formatted_prompt
        )
        print("----")
        print("---EXPE")
        print(expected)
        print("---")
        assert (
            task.generate_prompt(
                **mock_kwargs, evolution_method=evolution_method
            ).formatted_prompt
            == expected
        )
    else:
        with pytest.raises(expected):
            task.generate_prompt(**mock_kwargs, evolution_method=evolution_method)


def get_custom_evol_dataset(
    instruction_type: object,
) -> CustomDataset:
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
            "instructions": [
                [
                    'Compose a concise and articulate sentence that incorporates the terms "ecstatic," "exhilarated," "blissful," and "overjoyed."'
                ],
                [
                    "Design comprehensive and detailed plumbing diagrams for a two-story house that include separate diagrams for each floor, showcasing the layout, dimensions, and connections of all plumbing fixtures, pipelines, drains, vents, water supply sources, and sewage disposal systems. These diagrams should consider the specific requirements and codes of local building regulations while providing a clear and accurate representation of the plumbing infrastructure throughout the entire house."
                ],
            ],
        }
    )
    ds.task = instruction_type()
    return ds


@pytest.mark.parametrize(
    "instruction_type",
    [
        EvolInstructTask,
        EvolComplexityTask,
    ],
)
def test_evol_task_to_argilla_dataset(
    instruction_type,
):
    ds = get_custom_evol_dataset(instruction_type)
    ds_row = ds[0]
    task = ds.task
    rg_dataset = task.to_argilla_dataset(ds_row)
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 0


@pytest.mark.parametrize(
    "instruction_type",
    [
        EvolInstructTask,
        EvolComplexityTask,
    ],
)
def test_evol_task_to_argilla_record(
    instruction_type,
):
    ds: CustomDataset = get_custom_evol_dataset(instruction_type)
    ds_row = ds[0]
    task = ds.task
    records = task.to_argilla_record(ds_row)
    assert isinstance(records, list)
    assert len(records) == 1
    assert isinstance(records[0], rg.FeedbackRecord)


@pytest.mark.parametrize(
    "instruction_type",
    [
        EvolInstructTask,
        EvolComplexityTask,
    ],
)
def test_evol_task_to_argilla(
    instruction_type,
):
    ds = get_custom_evol_dataset(instruction_type)
    rg_dataset = ds.to_argilla(vector_strategy=False, metric_strategy=False)
    assert isinstance(rg_dataset, rg.FeedbackDataset)
    assert len(rg_dataset.records) == 2
