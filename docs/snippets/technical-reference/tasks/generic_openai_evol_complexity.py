import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import EvolInstructTask

generator = OpenAILLM(
    task=EvolInstructTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
