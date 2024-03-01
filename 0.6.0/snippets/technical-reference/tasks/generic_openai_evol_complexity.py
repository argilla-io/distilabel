import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import EvolComplexityTask

generator = OpenAILLM(
    task=EvolComplexityTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
