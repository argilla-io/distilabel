import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import EvolComplexityScorerTask

labeller = OpenAILLM(
    task=EvolComplexityScorerTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
