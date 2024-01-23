from dataclasses import dataclass
import re
from typing import Any, List, Dict

from distilabel.tasks import CritiqueTask, CritiqueTaskOutput
from distilabel.tasks.prompt import Prompt

input_prompt = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{{ instruction }}

###Response to evaluate:
{{ completion }}

###Reference Answer (Score 5):
{{ ref_completion }}

###Score Rubrics:
[{{ scoring_criteria }}]
Score 1: {{ score_descriptions[0] }}
Score 2: {{ score_descriptions[1] }}
Score 3: {{ score_descriptions[2] }}
Score 4: {{ score_descriptions[3] }}
Score 5: {{ score_descriptions[4] }}

###Feedback: 
"""


@dataclass
class CustomCritique(CritiqueTask):
    system_prompt: str = "You are a fair evaluator language model."

    def generate_prompt(
        self, input: str, generations: str, ref_completion: str, **_: Any
    ) -> Prompt:
        render_kwargs = {
            "instruction": input,
            "completion": generations,
            "ref_completion": ref_completion,
            "scoring_criteria": self.scoring_criteria,
            "score_descriptions": self.score_descriptions,
        }

        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> List[Dict]:
        """Parses the output of the model into the desired format."""
        pattern = r"(.+?)\. \[RESULT\] (\d+)"
        match = re.search(pattern, output)
        if match:
            return CritiqueTaskOutput(
                score=float(match.group(2)),
                critique=match.group(1).strip(),
            )
