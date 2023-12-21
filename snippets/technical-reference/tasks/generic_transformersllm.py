from distilabel.llm import TransformersLLM
from distilabel.tasks import TextGenerationTask

# This snippet uses `TransformersLLM`, but is the same for every other `LLM`.
generator = TransformersLLM(
    model=...,
    tokenizer=...,
    task=TextGenerationTask(),
)
