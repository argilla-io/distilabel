import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import UltraFeedbackTask

labeller = OpenAILLM(
    task=UltraFeedbackTask.for_overall_quality(),
    api_key=os.getenv("OPENAI_API_KEY", None),
)
