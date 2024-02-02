import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import JudgeLMTask

labeller = OpenAILLM(task=JudgeLMTask(), api_key=os.getenv("OPENAI_API_KEY", None))
