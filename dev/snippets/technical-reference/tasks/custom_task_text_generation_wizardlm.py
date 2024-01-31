from dataclasses import dataclass
import string
from typing import Dict, List

from distilabel.tasks import Prompt, TextGenerationTask

# Prompt from the WizardLM paper for the Equal Prompts task:
wizardllm_equal_prompt = """Here are two Instructions, do you think they are equal to each other and meet the following requirements?:
1. They have the same constraints and requirments.
2. They have the same depth and breadth of the inquiry.
The First Prompt: {first_instruction}
The Second Prompt: {second_instruction}
Your Judgement (Just answer: Equal or Not Equal. No need to explain the reason):"""


@dataclass
class WizardLMEqualPrompts(TextGenerationTask):
    """Task to check for the equality of two instructions following the Appendix G in
    [WizardLM paper](https://arxiv.org/abs/2304.12244).
    """

    system_prompt: str = "You are an AI judge in charge of determining the equality of two instructions. "

    def generate_prompt(self, input: List[str]) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=wizardllm_equal_prompt.format(
                first_instruction=input[0], second_instruction=input[1]
            ),
        )

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        """Remove punctuation from the string."""
        return {
            "generations": output.translate(str.maketrans("", "", string.punctuation))
        }
