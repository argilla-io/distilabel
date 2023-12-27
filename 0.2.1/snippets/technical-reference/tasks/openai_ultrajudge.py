import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import UltraJudgeTask

labeller = OpenAILLM(task=UltraJudgeTask(), openai_api_key=os.getenv("OPENAI_API_KEY"))
