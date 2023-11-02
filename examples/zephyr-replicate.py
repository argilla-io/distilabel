import os

from datasets import load_dataset
from ultralabel.llm.openai_ import OpenAILLM
from ultralabel.llm.replicate import ReplicateLLM
from ultralabel.pipeline import Pipeline
from ultralabel.tasks.preference.ultrafeedback import MultiRatingTask
from ultralabel.tasks.text_generation.llama import Llama2GenerationTask

os.environ["REPLICATE_API_TOKEN"] = "r..."
os.environ["OPENAI_API_KEY"] = "sk-..."

llm = ReplicateLLM(
    endpoint_url="nateraw/zephyr-7b-beta:b79f33de5c6c4e34087d44eaea4a9d98ce5d3f3a09522f7328eea0685003a931",
    task=Llama2GenerationTask(),
    temperature=1.0,
    max_new_tokens=128,
    num_threads=4,
)

pipeline = Pipeline(
    generator=llm,
    labeller=OpenAILLM(
        model="gpt-3.5-turbo",
        task=MultiRatingTask.for_text_quality(),
        max_new_tokens=128,
        num_threads=2,
        temperature=0.0,
    ),
)

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "instruction")
    .select(range(10))
)

dataset = pipeline.generate(
    dataset, num_generations=2, batch_size=3, display_progress_bar=True
)
