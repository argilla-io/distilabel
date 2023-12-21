import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import OpenAITextGenerationTask

generator = OpenAILLM(
    task=OpenAITextGenerationTask(), openai_api_key=os.getenv("OPENAI_API_KEY")
)
