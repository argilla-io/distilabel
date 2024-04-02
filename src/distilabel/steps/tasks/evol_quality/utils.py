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

from typing import Dict

REWRITE_INSTRUCTION = """
I want you to act as a Response Rewriter.
Your goal is to enhance the quality of the response given by an AI assistant to the #Given Prompt# through rewriting.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt# and #Given Response#. Also, please do not omit the input in #Given Prompt#.\n
You Should enhance the quality of the response using the following method: \n{}
You should try your best not to make the #Rewritten Response# become verbose, #Rewritten Response# can only add 10 to 20 words into #Given Response#.
'#Given Response#', '#Rewritten Response#', 'given response' and 'rewritten response' are not allowed to appear in #Rewritten Response#
#Given Prompt#:
<PROMPT>
#Given Response#:
<RESPONSE>
#Rewritten Response#:
""".lstrip()

MUTATION_TEMPLATES: Dict[str, str] = {
    "HELPFULNESS": REWRITE_INSTRUCTION.format(
        "Please make the Response more helpful to the user."
    ),
    "RELEVANCE": REWRITE_INSTRUCTION.format(
        "Please make the Response more relevant to #Given Prompt#."
    ),
    "DEEPENING": REWRITE_INSTRUCTION.format("Please make the Response more in-depth."),
    "CREATIVITY": REWRITE_INSTRUCTION.format(
        "Please increase the creativity of the response."
    ),
    "DETAILS": REWRITE_INSTRUCTION.format(
        "Please increase the detail level of Response."
    ),
}
