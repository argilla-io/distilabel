from dataclasses import dataclass
from typing import Dict, List

from distilabel.tasks import SelfInstructTask


@dataclass
class CustomTask(SelfInstructTask):
    system_prompt: str = "You are an expert Haiku writer, writing the best and most diverse Haikus given topics as inputs."
    application_description: str = (
        "An AI assistant adept at writing Haiku.\n"
        "It expects complete suggestions from users providing details of the kind of haiku they want.\n"
        "The AI assistant will help users write haiku about particular topics and is willing to accept requests related to a specific subject or object or a more abstract request"
        "based on an emotion, theme or vibe.\n"
    )

    criteria_queries: str = (
        "Incorporate a diverse range of verbs, avoiding repetition.\n"
        "Ensure queries are compatible with AI model's text generation functions and are limited to 1-2 sentences.\n"
        "Design queries to be self-contained and standalone."
    )

    def define_task(self):
        instruction_task = SelfInstructTask(
            num_instructions=15,
            application_description=self.application_description,
            criteria_for_query_generation=self.criteria_queries,
        )

        return instruction_task

    def parse_output(self, output: str) -> List[Dict[str, str]]:
        return {"output": output}
