from dataclasses import dataclass
from typing import Dict, List

from distilabel.tasks import TextGenerationTask
from distilabel.tasks.prompt import Prompt

@dataclass
class CustomTask(TextGenerationTask):
    system_prompt: str = "You are a text-generation assistant for ...."
    input_prompt: str = "Please, generate text given ..."

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input_prompt
          )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        return {
            "output": output
        }
