from typing import Dict, List

from distilabel.tasks import Prompt, Task


class CustomTask(Task):
    @property
    def input_args_names(self) -> List[str]:
        return ["input", "generation"]

    @property
    def output_args_names(self) -> List[str]:
        return ["score"]

    def generate_prompt(self, input: str, generation: str) -> Prompt:
        return Prompt(
            system_prompt="You're a helpful AI assistant...",
            formatted_prompt=(
                "Given the following instruction and text generation, evaluate how good"
                " the generated text is giving a score between 0 and 1. Just return the score.\n\n"
                f"### Instruction\n{input}\n\n"
                f"### Generation\n{generation}"
            ),
        )

    def parse_output(self, output: str) -> Dict[str, float]:
        extracted_score = float(output)
        return {"score": extracted_score}


custom_task = CustomTask()
prompt = custom_task.generate_prompt("What's 2 + 2?", "2 + 2 is 5")
print(prompt.format_as("default"))
# You're a helpful AI assistant...
# Given the following instruction and text generation, evaluate how good the generated text is giving a score between 0 and 1. Just return the score.
#
# ### Instruction
# What's 2 + 2?
#
# ### Generation
# 2 + 2 is 5

print(custom_task.parse_output("0.0"))
# {"score": 0.0}
