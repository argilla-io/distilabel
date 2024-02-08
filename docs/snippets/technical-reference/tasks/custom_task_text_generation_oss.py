from dataclasses import dataclass
from typing import Dict, List

from distilabel.tasks import TextGenerationTask
from distilabel.tasks.prompt import Prompt

oss_instruct_prompt = """Please gain inspiration from the following random code snippet to create a high-quality programming problem. Present your output in two distinct sections:
[Problem Description] and [Solution].
Code snippet for inspiration:

{code}

Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing
all the contextual information one needs to understand and solve the problem.
Assume common programming knowledge, but ensure that any specific context,
variables, or code snippets pertinent to this problem are explicitly included.
2. [Solution]: Offer a comprehensive, **correct** solution that accurately
addresses the [Problem Description] you provided.
"""


@dataclass
class OSSInstruct(TextGenerationTask):
    system_prompt: str = "You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions."

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=oss_instruct_prompt.format(code=input)
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        problem, solution = output.split("[Solution]")
        return {
            "problem": problem.replace("[Problem Description]", "").strip(),
            "solution": solution.strip()
        }