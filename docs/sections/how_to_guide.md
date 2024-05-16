# How to Guide

To start off, `distilabel` is a framework for building pipelines for generating synthetic data using LLMs, that defines a [`Pipeline`][distilabel.pipeline.Pipeline] which orchestrates the execution of the [`Step`][distilabel.steps.base.Step] subclasses, and those will be connected as nodes in a Direct Acyclic Graph (DAG).

This being said, in this guide we will walk you through the process of creating a simple pipeline that uses the [`OpenAILLM`][distilabel.llms.OpenAILLM] class to generate text.å The [`Pipeline`][distilabel.pipeline.Pipeline] will load a dataset that contains a column named `prompt` from the Hugging Face Hub via the step [`LoadHubDataset`][distilabel.steps.LoadHubDataset] and then use the [`OpenAILLM`][distilabel.llms.OpenAILLM] class to generate text based on the dataset using the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task.

```python
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadHubDataset
from distilabel.steps.tasks import TextGeneration

with Pipeline(  # (1)
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:  # (2)
    load_dataset = LoadHubDataset(  # (3)
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(  # (4)
        name="text_generation",
        llm=OpenAILLM(model="gpt-3.5-turbo"),  # (5)
    )

    load_dataset >> text_generation  # (6)

if __name__ == "__main__":
    distiset = pipeline.run(  # (7)
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
    distiset.push_to_hub(repo_id="distilabel-example")  # (8)
```

1. We define a [`Pipeline`][distilabel.pipeline.Pipeline] with the name `simple-text-generation-pipeline` and a description `A simple text generation pipeline`. Note that the `name` is mandatory and will be used to calculate the `cache` signature path, so changing the name will change the cache path and will be identified as a different pipeline.

2. We are using the [`Pipeline`][distilabel.pipeline.Pipeline] context manager, meaning that every [`Step`][distilabel.steps.base.Step] subclass that is defined within the context manager will be added to the pipeline automatically.

3. We define a [`LoadHubDataset`][distilabel.steps.LoadHubDataset] step named `load_dataset` that will load a dataset from the Hugging Face Hub, as provided via runtime parameters in the `pipeline.run` method below, but it can also be defined within the class instance via the arg `repo_id=...`. This step will basically produce output batches with the rows from the dataset, and the column `prompt` will be mapped to the `instruction` field.

4. We define a [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task named `text_generation` that will generate text based on the `instruction` field from the dataset. This task will use the [`OpenAILLM`][distilabel.llms.OpenAILLM] class with the model `gpt-3.5-turbo`.

5. We define the [`OpenAILLM`][distilabel.llms.OpenAILLM] class with the model `gpt-3.5-turbo` that will be used by the [`TextGeneration`][distilabel.steps.tasks.TextGeneration] task. In this case, since the [`OpenAILLM`][distilabel.llms.OpenAILLM] is used, we assume that the `OPENAI_API_KEY` environment variable is set, and the OpenAI API will be used to generate the text.

6. We connect the `load_dataset` step to the `text_generation` task using the `rshift` operator, meaning that the output from the `load_dataset` step will be used as input for the `text_generation` task.

7. We run the pipeline with the parameters for the `load_dataset` and `text_generation` steps. The `load_dataset` step will use the repository `distilabel-internal-testing/instruction-dataset-mini` and the `test` split, and the `text_generation` task will use the `generation_kwargs` with the `temperature` set to `0.7` and the `max_new_tokens` set to `512`.

8. Optionally, we can push the generated [`Distiset`][distilabel.distiset.Distiset] to the Hugging Face Hub repository `distilabel-example`. This will allow you to share the generated dataset with others and use it in other pipelines.
