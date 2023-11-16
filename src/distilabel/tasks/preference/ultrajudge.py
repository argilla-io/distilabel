import re
from dataclasses import dataclass, field
from typing import Dict, List, TypedDict

from distilabel.tasks.base import Prompt, Task, get_template

_ULTRAJUDGE_TEMPLATE = get_template("ultrajudge.jinja2")


class Area(TypedDict):
    rating: int
    rationale: str


class UltraJudgeOutput(TypedDict):
    rating: int
    areas: Dict[str, Area]


@dataclass
class UltraJudgeTask(Task):
    system_prompt: str = (
        "You are an evaluator tasked with assessing AI assistants' responses from the perspective of typical user preferences."
        " Your critical analysis should focus on human-like engagement, solution effectiveness, accuracy, clarity, and"
        " creativity. Approach each response as if you were the user, considering how well the response meets your needs"
        " and expectations in a real-world scenario. Provide detailed feedback that highlights strengths and areas for"
        " improvement in each response, keeping in mind the goal of simulating a human's preferred choice. "
        "Your evaluation should be impartial and thorough, reflecting a human's perspective in preferring responses that are practical,"
        " clear, authentic, and aligned with their intent. Avoid bias, and focus on the content and quality of the responses."
    )

    task_description: str = (
        "Your task is to rigorously evaluate the performance of {num_responses} AI assistants, simulating a human's perspective."
        " You will assess each response based on four key domains, reflecting aspects that are typically valued by humans:"
        " {areas}."
        " First provide a score between 0 and 10 and write a detailed feedback for each area and assistant."
        " Finally, provide a list of {num_responses} scores, each separated by a space, to reflect the performance of Assistants 1 to {num_responses}."
    )

    areas: List[str] = field(
        default_factory=lambda: [
            "Practical Accuracy",
            "Clarity & Transparency",
            "Authenticity & Reliability",
            "Compliance with Intent",
        ]
    )

    __jinja2_template__: str = field(
        default=_ULTRAJUDGE_TEMPLATE, init=False, repr=False
    )

    @property
    def input_args_names(self) -> List[str]:
        return ["input", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["rating", "rationale"]

    @property
    def areas_str(self) -> str:
        return ", ".join(self.areas[:-1]) + ", and " + self.areas[-1]

    @property
    def extract_area_score_and_rationale_regex(self) -> str:
        return rf"({'|'.join(self.areas)})\s*-\s*(\d+(?:\.\d+)?)\n(.*?)(?=\n\n|\Z)"

    @property
    def extract_final_scores_regex(self) -> str:
        return r"Final scores:\s*((?:\d+(?:\.\d+)?\s*)+)"

    def generate_prompt(self, input: str, generations: List[str]) -> Prompt:
        render_kwargs = {
            "task_description": self.task_description.format(
                num_responses=len(generations), areas=self.areas_str
            ),
            "instruction": input,
            "responses": generations,
        }

        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> List[UltraJudgeOutput]:
        num_areas = len(self.areas)
        # `areas_results` includes num_generations * num_areas tuples
        areas_results = re.findall(self.extract_area_score_and_rationale_regex, output)
        final_scores = [
            int(float(str_score))
            for str_score in re.findall(self.extract_final_scores_regex, output)[
                0
            ].split(" ")
        ]

        outputs = []
        for i, rating in enumerate(final_scores):
            areas = {}
            # Get the areas for the i-th generation
            for area in areas_results[i * num_areas : i * num_areas + num_areas]:
                name, area_rating, rationale = area
                areas[name] = Area(rating=area_rating, rationale=rationale)
            outputs.append(UltraJudgeOutput(rating=rating, areas=areas))

        return outputs
