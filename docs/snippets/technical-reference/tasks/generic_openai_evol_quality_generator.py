import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import EvolQualityGeneratorTask

generator = OpenAILLM(
    task=EvolQualityGeneratorTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
