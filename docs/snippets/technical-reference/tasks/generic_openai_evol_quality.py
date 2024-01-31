import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import EvolQualityTask

generator = OpenAILLM(
    task=EvolQualityTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
