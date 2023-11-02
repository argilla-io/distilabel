import time
from uuid import uuid4

import argilla as rg
from datasets import load_dataset
from ultralabel.llm.huggingface import InferenceEndpointsLLM
from ultralabel.llm.openai_ import OpenAILLM
from ultralabel.pipeline import Pipeline
from ultralabel.tasks.preference.ultrafeedback import PreferenceTask
from ultralabel.tasks.text_generation.llama import Llama2GenerationTask

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:10]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "instruction")
)

pipeline = Pipeline(
    generation_llm=InferenceEndpointsLLM(
        endpoint_url="<INFERENCE_ENDPOINT_URL>",
        task=Llama2GenerationTask(),
        max_new_tokens=128,
        num_threads=4,
        temperature=0.3,
    ),
    labelling_llm=OpenAILLM(
        model="gpt-3.5-turbo",
        task=PreferenceTask.for_text_quality(),
        max_new_tokens=128,
        num_threads=2,
        openai_api_key="<OPENAI_API_KEY>",
        temperature=0.0,
    ),
)

start = time.time()
dataset = pipeline.generate(
    dataset, num_generations=2, batch_size=1, display_progress_bar=True
)
end = time.time()
print("Elapsed", end - start)

# Push to the HuggingFace Hub
dataset.push_to_hub("<REPO_ID>", split="train", private=True)

# Convert into an Argilla dataset and push it to Argilla
rg.init(api_url="<ARGILLA_API_URL>", api_key="<ARGILLA_API_KEY>")
rg_dataset = dataset.to_argilla()
rg_dataset.push_to_argilla(name=f"my-dataset-{uuid4()}", workspace="admin")
