from typing import TYPE_CHECKING, Dict, List

from ultralabel.tasks.base import Task, get_template
from ultralabel.tasks.utils import Prompt

if TYPE_CHECKING:
    from ultralabel.tasks.utils import ChatCompletion

_SELF_INSTRUCT_TEMPLATE = get_template("self-instruct.jinja2")


class OpenAITextGenerationTask(Task):
    system_prompt: str = (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,"
        " while being safe. Your answers should not include any harmful, unethical, racist, sexist,"
        " toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased"
        " and positive in nature.\nIf a question does not make any sense, or is not factually coherent,"
        " explain why instead of answering something not correct. If you don't know the answer to a"
        " question, please don't share false information."
    )

    def generate_prompt(self, instruction: str) -> List["ChatCompletion"]:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=instruction,
        ).format_as("openai")

    def parse_output(self, output: str) -> Dict[str, str]:
        return {"generations": output}

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction"]

    @property
    def output_args_names(self) -> List[str]:
        return ["generations"]


class SelfInstructTask(OpenAITextGenerationTask):
    __jinja2_template__: str = _SELF_INSTRUCT_TEMPLATE
    system_prompt: str = (
        "You are an expert prompt writer, writing the best and most diverse prompts for a variety of tasks."
        "You are given a task description and a set of instructions for how to write the prompts for a specific AI application."
    )

    application_description: str = "AI assistant"

    num_instructions: int = 5

    def generate_prompt(self, instruction: str) -> Prompt:
        render_kwargs = {
            "application_description": self.application_description,
            "num_instructions": self.num_instructions,
            "instruction": instruction,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )

    def parse_output(self, output: str) -> Dict[str, str]:
        parsed_output = output.split("\n")
        return {"generations": parsed_output}

    @property
    def input_args_names(self) -> List[str]:
        return ["instruction"]

    @property
    def output_args_names(self) -> List[str]:
        return ["generations"]
