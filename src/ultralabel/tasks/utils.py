from typing import List, Literal, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class ChatCompletion(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


# TODO: add more output formats
# TODO: move this file outside as `prompt.py` or something more meaningful
class Prompt(BaseModel):
    system_prompt: str
    formatted_prompt: str

    def format_as(
        self, format: Literal["openai", "llama2"]
    ) -> Union[str, List[ChatCompletion]]:
        if format == "openai":
            return [
                ChatCompletion(
                    role="system",
                    content=self.system_prompt,
                ),
                ChatCompletion(role="user", content=self.formatted_prompt),
            ]
        elif format == "llama2":
            return f"<s>[INST] <<SYS>>\n{self.system_prompt}<</SYS>>\n\n{self.formatted_prompt} [/INST]"
        else:
            raise ValueError(
                f"Format {format} not supported, please provide a custom `formatting_fn`."
            )
