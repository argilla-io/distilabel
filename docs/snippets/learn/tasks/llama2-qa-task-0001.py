from typing import Dict

from distilabel.tasks import Llama2TextGenerationTask, Prompt

class Llama2QuestionAnsweringTask(Llama2TextGenerationTask):
    def generate_prompt(self, question: str) -> str:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=question,
        ).format_as("llama2")  # type: ignore

    def parse_output(self, output: str) -> Dict[str, str]:
        return {"answer": output.strip()}

    def input_args_names(self) -> list[str]:
        return ["question"]

    def output_args_names(self) -> list[str]:
        return ["answer"]