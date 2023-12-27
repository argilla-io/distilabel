from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import pipeline
from distilabel.tasks import TextGenerationTask

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

task = TextGenerationTask()  # (1)

generator = OpenAILLM(task=task, max_new_tokens=512)  # (2)

pipeline = pipeline("preference", "instruction-following", generator=generator)  # (3)

dataset = pipeline.generate(dataset)
