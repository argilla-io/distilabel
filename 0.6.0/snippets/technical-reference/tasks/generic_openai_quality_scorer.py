import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import QualityScorerTask

labeller = OpenAILLM(
    task=QualityScorerTask(),
    api_key=os.getenv("OPENAI_API_KEY", None),
)
