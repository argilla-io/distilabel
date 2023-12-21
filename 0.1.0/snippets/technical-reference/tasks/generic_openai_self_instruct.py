import os

from distilabel.llm import OpenAILLM
from distilabel.tasks import SelfInstructTask

generator = OpenAILLM(
    task=SelfInstructTask(
        system_prompt="You are a question-answering assistant for...",
        application_description="AI assistant",
        num_instructions=3,
    ),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
