import argilla as rg
from datasets import load_dataset
from distilabel.llm.huggingface.inference_endpoints import InferenceEndpointsLLM
from distilabel.llm.openai_ import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks.preference.ultrafeedback import UltraFeedbackTask
from distilabel.tasks.text_generation.llama import Llama2TextGenerationTask

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:5]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "input")
)

pipeline = Pipeline(
    generator=InferenceEndpointsLLM(
        endpoint_name="<HUGGING_FACE_INFERENCE_ENDPOINT_NAME>",
        token="<HUGGING_FACE_HUB_TOKEN>",
        task=Llama2TextGenerationTask(),
        max_new_tokens=128,
        num_threads=4,
        temperature=0.3,
    ),
    labeller=OpenAILLM(
        model="gpt-3.5-turbo",
        task=UltraFeedbackTask.for_text_quality(),
        max_new_tokens=128,
        num_threads=2,
        openai_api_key="<OPENAI_API_KEY>",
        temperature=0.0,
    ),
)

dataset = pipeline.generate(
    dataset,  # type: ignore
    num_generations=2,
    batch_size=1,
    enable_checkpoints=True,
    display_progress_bar=True,
)

# Push to the HuggingFace Hub
dataset.push_to_hub("<REPO_ID>", split="train", private=True)

# Convert into an Argilla dataset and push it to Argilla
rg.init(api_url="<ARGILLA_API_URL>", api_key="<ARGILLA_API_KEY>")
rg_dataset = dataset.to_argilla()
rg_dataset.push_to_argilla(name="my-dataset", workspace="my-workspace")
