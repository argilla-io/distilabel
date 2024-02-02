import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import UltraFeedbackTask

labeller = OpenAILLM(
    task=UltraFeedbackTask.for_instruction_following(),
    api_key=os.getenv("OPENAI_API_KEY", None),
)
