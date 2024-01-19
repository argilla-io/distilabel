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

import pytest
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
def test_evol_instruct_task(self_instruct: SelfInstructTask, expected: str):
    assert isinstance(self_instruct, SelfInstructTask)
    assert (
        self_instruct.system_prompt
        == "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks. You are given a task description and a set of instructions for how to write the prompts for an specific AI application."
    )
    assert self_instruct.generate_prompt(input="HELLO").formatted_prompt == expected
