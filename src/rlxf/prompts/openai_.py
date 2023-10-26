from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Type, Union

from typing_extensions import TypedDict

from rlxf.prompts.base import PromptTemplate, get_template

try:
    import argilla as rg

    _argilla_installed = True
except ImportError:
    _argilla_installed = False

if TYPE_CHECKING:
    from argilla.client.feedback.schemas.records import FeedbackRecord
    from argilla.client.feedback.schemas.types import (
        AllowedFieldTypes,
        AllowedQuestionTypes,
    )

_GPT4_RANKING_TEMPLATE = get_template("gpt4-response-ranking.jinja2")
_GPT_TEXT_GENERATION_TEMPLATE = get_template("gpt-text-generation.jinja2")


class Rank(TypedDict):
    rank: int
    description: str


class RankOutput(TypedDict):
    rating: int
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
        render_kwargs = {
            "task_description": self.task_description,
            "ranks": self.ranks,
            "ranks_description": self.ranks_description,
            "instruction": instruction,
            "responses": generations,
        }
        generated_prompt = self.template.render(render_kwargs)
        return [
            ChatCompletion(
                role="system",
                content=self.system_prompt,
            ),
            ChatCompletion(role="user", content=generated_prompt),
        ]

    def _parse_output(self, output: str) -> List[RankOutput]:
        parsed_output = []
        for section in output.split("#### Output for Text ")[1:]:
            rating, rationale = section.split("\n")[1:3]
            rating = int(rating.split(": ")[1])
            rationale = rationale.split(": ")[1]
            parsed_output.append(RankOutput(rating=rating, rationale=rationale))
        return parsed_output

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction", "generations"]

    @property
    def output_args_names(self) -> List[str]:
        return ["rating", "rationale"]

    @property
    def argilla_fields_typedargs(self) -> Dict[str, Union[Type[str], Type[list]]]:
        # If a `List[str]` is provided, then it means that the field will be generated
        # appending an integer from 1 to N e.g. `generations` -> `generations-1`, `generations-2`, etc.
        return {"instruction": str, "generations": list}

    def to_argilla_fields(
        self, dataset_row: Dict[str, Any]
    ) -> List["AllowedFieldTypes"]:
        argilla_fields = []
        for arg_name, arg_type in self.argilla_fields_typedargs.items():
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if arg_type is list and isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    argilla_fields.append(
                        rg.TextField(
                            name=f"{arg_name}-{idx}", title=f"OpenAI Generation - {idx}"
                        )
                    )
            elif arg_type is str:
                argilla_fields.append(rg.TextField(name=arg_name))
            else:
                raise ValueError(f"Type {arg_type} is not supported.")
        return argilla_fields

    @property
    def argilla_questions_typedargs(self) -> Dict[str, Type[list]]:
        return {"generations": list}

    def to_argilla_questions(
        self, dataset_row: Dict[str, Any]
    ) -> List["AllowedQuestionTypes"]:
        if not _argilla_installed:
            raise ImportError("The argilla library is not installed.")
        argilla_questions = []
        for arg_name, arg_type in self.argilla_questions_typedargs.items():
            if arg_name not in dataset_row:
                raise ValueError(
                    f"Dataset row does not contain the required field '{arg_name}'."
                )
            if arg_type is list and isinstance(dataset_row[arg_name], list):
                for idx in range(1, len(dataset_row[arg_name]) + 1):
                    argilla_questions.extend(
                        [
                            rg.RatingQuestion(
                                name=f"generations-{idx}-rating",
                                title=f"Whats's the rating for the Generation - {idx}?",
                                values=list(range(1, len(self.ranks) + 1)),
                            ),
                            rg.TextQuestion(
                                name=f"generations-{idx}-rationale",
                                title=f"Whats's the rationale behind Generation - {idx} rating?",
                            ),
                        ]
                    )
        return argilla_questions

    def to_argilla_record(self, dataset_row: Dict[str, Any]) -> "FeedbackRecord":
        fields = {}
        for input_arg_key, input_arg_value in self.argilla_fields_typedargs.items():
            if input_arg_value is list:
                for idx in range(1, len(dataset_row[input_arg_key]) + 1):
                    fields.update(
                        {f"{input_arg_key}-{idx}": dataset_row[input_arg_key][idx - 1]}
                    )
            else:
                fields.update({input_arg_key: dataset_row[input_arg_key]})
        response = {"values": {}, "status": "submitted"}
        for output_arg_name in self.output_args_names:
            for idx, value in enumerate(dataset_row[output_arg_name]):
                response["values"].update(
                    {f"generations-{idx}-{output_arg_name}": {"value": value}}
                )
        return rg.FeedbackRecord(fields=fields, responses=[response])


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
        generated_prompt = self.template.render(
            system_prompt=self.system_prompt, instruction=instruction
        )
        return [
            ChatCompletion(
                role="system",
                content=self.system_prompt,
            ),
            ChatCompletion(role="user", content=generated_prompt),
        ]

    def parse_output(self, output: str) -> Dict[str, str]:
        return {"generations": output}

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction"]

    @property
    def output_args_names(self) -> List[str]:
        return ["generations"]
