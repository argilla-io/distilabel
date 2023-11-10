# distilabel

AI Feedback framework to build datasets with and for LLMs

## Installation

```sh
pip install distilabel
```

# Quickstart examples

## Instruction dataset generation

```py title="Generate instructions for table generation about different topics"
from datasets import Dataset

from distilabel.llm import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import SelfInstructTask

self_instruct = SelfInstructTask(
    application_description="An AI application to generate tables",
    num_instructions=10
)

llm = OpenAILLM(
    model="gpt-4",
    task=self_instruct,
    max_new_tokens=1024,
    num_threads=2,
    token="sk..."
)

dataset = Dataset.from_dict(
    {"instruction": ["High school math", "Celebrities", "Astrophysics"]}
)

pipeline = Pipeline(
    generator=llm
)

dataset = pipeline.generate(
    dataset, num_generations=1, batch_size=1, display_progress_bar=True
)
```
## Preference dataset generation

```py title="Build a preference dataset based on response's truthfulness"
from datasets import load_dataset
from distilabel.pipeline import pipeline
from distilabel.llm import InferenceEndpointsLLM
from distilabel.tasks.text_generation import Llama2GenerationTask

dataset = (
    load_dataset("HuggingFaceH4/instruction-dataset", split="test[:5]")
    .remove_columns(["completion", "meta"])
    .rename_column("prompt", "instruction")
)

llm = InferenceEndpointsLLM(
        endpoint_url="<INFERENCE_ENDPOINTS_URL>",
        task=Llama2GenerationTask(), 
        max_new_tokens=128,
        num_threads=4,
        temperature=0.3,
)

pipe = pipeline(
    "preference",
    "truthfulness",
    generator=llm
)
```
