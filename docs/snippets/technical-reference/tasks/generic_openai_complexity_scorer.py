import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import ComplexityScorerTask

labeller = OpenAILLM(
    task=ComplexityScorerTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
