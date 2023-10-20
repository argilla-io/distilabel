from dataclasses import dataclass
from typing import List

import jinja2
import pkg_resources  # https://setuptools.pypa.io/en/latest/pkg_resources.html

from rlxf.prompts.ranking import RankingPromptTemplate, RankOutput

_GPT4_RANKING_TEMPLATE = pkg_resources.resource_filename(
    # "rlxf", "prompts/templates/files/gpt4/ranking.jinja2"
    "rlxf",
    "prompts/templates/files/prometheus.jinja2",
)


@dataclass
class GPT4RankingPromptTemplate(RankingPromptTemplate):
    __jinja2_template__: str = _GPT4_RANKING_TEMPLATE

    def generate_prompt(self, instruction: str, responses: List[str]) -> str:
        template = jinja2.Template(open(self.__jinja2_template__).read())
        return template.render(
            system_prompt=self.system_prompt,
            ranks=self.ranks,
            ranks_description=self.ranks_description,
            instruction=instruction,
            response=responses[0],
        )

    def parse_output(self, output: str) -> RankOutput:
        parts = output.split("[RESULT]")
        return RankOutput(score=int(parts[0].strip()), rationale=parts[1].strip())


if __name__ == "__main__":
    prompt_template = GPT4RankingPromptTemplate(
        system_prompt="You are a teacher grading a student's answer.",
        ranks=[
            {"rank": 1, "description": "Incorrect"},
            {"rank": 2, "description": "Partially correct"},
            {"rank": 3, "description": "Correct"},
        ],
        ranks_description="Rank the following answers from best to worst.",
    )
    prompt = prompt_template.generate_prompt(
        instruction="What's the capital city of France?",
        responses=["Paris", "London", "New York"],
    )
    print(prompt)
