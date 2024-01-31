import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import QualityScorerTask

labeller = OpenAILLM(
    task=QualityScorerTask(task_description="Take into account the expressiveness of the answers."),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
