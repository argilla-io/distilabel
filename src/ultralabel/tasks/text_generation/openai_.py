from typing import TYPE_CHECKING, Dict, List

from ultralabel.tasks.base import Task
from ultralabel.tasks.utils import Prompt

if TYPE_CHECKING:
    from ultralabel.tasks.utils import ChatCompletion


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
