import importlib.resources as importlib_resources
from textwrap import dedent
from typing import List, Literal, Union

import jinja2
from typing_extensions import TypedDict

from rlxf.prompts.base import PromptTemplate

_GPT4_RANKING_TEMPLATE = str(
    importlib_resources.files("rlxf") / "prompts/templates/gpt4-response-ranking.jinja2"
)
_GPT_TEXT_GENERATION_TEMPLATE = str(
    importlib_resources.files("rlxf") / "prompts/templates/gpt-text-generation.jinja2"
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
        self, instruction: str, generations: List[str]
    ) -> Union[str, List[ChatCompletion]]:
        template = jinja2.Template(open(self.__jinja2_template__).read())
        render_kwargs = {
            "task_description": self.task_description,
            "ranks": self.ranks,
            "ranks_description": self.ranks_description,
            "instruction": instruction,
            "responses": generations,
        }
        generated_prompt = template.render(render_kwargs)
        return [
            ChatCompletion(
                role="system",
                content=self.system_prompt,
            ),
            ChatCompletion(role="user", content=generated_prompt),
        ]

    def parse_output(self, output: str) -> List[RankOutput]:
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = int(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append(RankOutput(score=rating, rationale=rationale))
        return parsed_output

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["score", "rationale"]


class OpenAITextGenerationPromptTemplate(PromptTemplate):
    __jinja2_template__: str = _GPT_TEXT_GENERATION_TEMPLATE

    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )

    def generate_prompt(self, instruction: str) -> Union[str, List[ChatCompletion]]:
        template = jinja2.Template(open(self.__jinja2_template__).read())
        generated_prompt = template.render(
            system_prompt=self.system_prompt, instruction=instruction
        )
        return [
            ChatCompletion(
                role="system",
                content=self.system_prompt,
            ),
            ChatCompletion(role="user", content=generated_prompt),
        ]

    def parse_output(self, output: str) -> str:
        return {"generations": output}

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction"]

    @property
    def output_args_names(self) -> List[str]:
        return ["generations"]
