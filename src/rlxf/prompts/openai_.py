import importlib.resources as importlib_resources
from textwrap import dedent
from typing import List, Literal

import jinja2
from typing_extensions import TypedDict

from rlxf.prompts.base import PromptTemplate

_GPT4_RANKING_TEMPLATE = (
    importlib_resources.files("rlxf")
    / "prompts/templates/files/gpt4/response-ranking.jinja2"
)


class Rank(TypedDict):
    rank: int
    description: str


class RankOutput(TypedDict):
    score: int
    rationale: str


class ChatCompletion(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIResponseRanking(PromptTemplate):
    task_description: str
    ranks: List[Rank]
    ranks_description: str

    __type__: str = "ranking"
    __jinja2_template__: str = _GPT4_RANKING_TEMPLATE

    system_prompt: str = (
        "Your role is to evaluate text quality based on given criteria."
    )
    task_description: str = dedent(
        """
        # Informativeness / Helpfulness Assessment

        Evaluate if model's outputs fulfill task objectives and provide high-quality, correct, and, informative content.

        Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativeness.

        **Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.
    """
    )

    def generate_prompt(
        self, instruction: str, responses: List[str], for_chat: bool = True
    ) -> str | List[ChatCompletion]:
        template = jinja2.Template(open(self.__jinja2_template__).read())
        render_kwargs = {
            "task_description": self.task_description,
            "ranks": self.ranks,
            "ranks_description": self.ranks_description,
            "instruction": instruction,
            "responses": responses,
        }
        if not for_chat:
            render_kwargs["system_prompt"] = self.system_prompt
        generated_prompt = template.render(render_kwargs)
        if not for_chat:
            return generated_prompt
        return [
            ChatCompletion(
                role="system",
                content=self.system_prompt,
            ),
            ChatCompletion(
                role="user",
                content=generated_prompt.replace(
                    r"\{\{ system_prompt \}\}", ""
                ).lstrip(),
            ),
        ]

    def parse_output(self, output: str) -> List[RankOutput]:
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = int(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append(RankOutput(score=rating, rationale=rationale))
        return parsed_output
