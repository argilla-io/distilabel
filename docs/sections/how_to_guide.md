# How to Guide

```python
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset
from distilabel.steps.tasks import TextGeneration

with Pipeline(
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:
    load_dataset = LoadHubDataset(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=OpenAILLM(model="gpt-3.5-turbo"),
    )

    load_dataset >> text_generation

if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                        "max_new_tokens": 512,
                    }
                }
            },
        },
    )
```
