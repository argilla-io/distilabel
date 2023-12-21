from distilabel.llm import TransformersLLM
from distilabel.tasks import Llama2TextGenerationTask

# This snippet uses `TransformersLLM`, but is the same for every other `LLM`.
generator = TransformersLLM(
    model=...,
    tokenizer=...,
    task=Llama2TextGenerationTask(),
)
