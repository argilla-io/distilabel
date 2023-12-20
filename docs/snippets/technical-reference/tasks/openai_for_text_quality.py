import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import UltraFeedbackTask

labeller = OpenAILLM(
    task=UltraFeedbackTask.for_text_quality(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
