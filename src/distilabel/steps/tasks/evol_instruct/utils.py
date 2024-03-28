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

REWRITE_INSTRUCTION = """
I want you act as a Prompt Rewriter.\n
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\n
But the rewritten prompt must be reasonable and must be understood and responded by humans.\n
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.\n
You SHOULD complicate the given prompt using the following method: \n{}\n
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\n
#The Given Prompt#:\n<PROMPT>\n#Rewritten Prompt#:\n
""".lstrip()

CREATE_INSTRUCTION = """
I want you act as a Prompt Creator.\n
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\n
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\n
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\n
The #Created Prompt# must be reasonable and must be understood and responded by humans.\n
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\n
#Given Prompt#:\n<PROMPT>\n#Created Prompt#:\n
""".lstrip()

MUTATION_TEMPLATES = {
    "CONSTRAINTS": REWRITE_INSTRUCTION.format(
        "Please add one more constraints/requirements into '#The Given Prompt#'"
    ),
    "DEEPENING": REWRITE_INSTRUCTION.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    ),
    "CONCRETIZING": REWRITE_INSTRUCTION.format(
        "Please replace general concepts with more specific concepts."
    ),
    "INCREASED_REASONING_STEPS": REWRITE_INSTRUCTION.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    ),
    "BREADTH": CREATE_INSTRUCTION,
}

GENERATION_MUTATION_TEMPLATES = {
    "FRESH_START": "Write one question or request containing one or more of the following words: <PROMPT>",
    "CONSTRAINTS": REWRITE_INSTRUCTION.format(
        "Please add one more constraints/requirements into '#The Given Prompt#'"
    ),
    "DEEPENING": REWRITE_INSTRUCTION.format(
        "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    ),
    "CONCRETIZING": REWRITE_INSTRUCTION.format(
        "Please replace general concepts with more specific concepts."
    ),
    "INCREASED_REASONING_STEPS": REWRITE_INSTRUCTION.format(
        "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    ),
    "BREADTH": CREATE_INSTRUCTION,
}
