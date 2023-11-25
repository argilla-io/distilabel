import os

import argilla as rg
from datasets import load_dataset
from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import JudgeLMTask

os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"
rg.init(api_url="<ARGILLA_API_KEY>", api_key="<ARGILLA_API_URL>")

dataset = load_dataset("gabrielmbmb/ultrafeedback-prompts-judgelm-gpt35", split="train")

dataset = dataset.remove_columns(  # .shuffle()
    [
        "generation_model",
        "generation_prompt",
        "raw_generation_responses",
        "labelling_model",
        "labelling_prompt",
        "raw_labelling_response",
        "ratings",
        "rationale",
    ]
).select(  # type: ignore
    range(1)
)

labeller = OpenAILLM(
    model="gpt-3.5-turbo",
    task=JudgeLMTask(),
    max_new_tokens=1024,
    num_threads=16,
    temperature=1.0,
)

pipeline = Pipeline(labeller=labeller)

labelled_dataset = pipeline.generate(
    dataset,  # type: ignore
    num_generations=2,
    batch_size=8,
    enable_checkpoints=True,
    display_progress_bar=True,
)

rg_dataset = labelled_dataset.to_argilla()
rg_dataset.push_to_argilla(
    name="distilabel-judgelm", workspace="<ARGILLA_WORKSPACE_NAME>"
)
