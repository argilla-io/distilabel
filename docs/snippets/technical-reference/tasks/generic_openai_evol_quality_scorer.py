import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import EvolQualityScorerTask

labeller = OpenAILLM(
    task=EvolQualityScorerTask(),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
