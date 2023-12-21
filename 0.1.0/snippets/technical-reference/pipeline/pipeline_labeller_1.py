import os

from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import UltraFeedbackTask

pipe_labeller = Pipeline(
    labeller=OpenAILLM(
        model="gpt-4",
        task=UltraFeedbackTask.for_instruction_following(),
        max_new_tokens=256,
        num_threads=8,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    ),
)
