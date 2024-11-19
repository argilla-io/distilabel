---
description: Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
hide:
  - toc
---

<a target="_blank" href="https://colab.research.google.com/drive/1DJFDZtOfnNYg7ZfmZPfICm750tuJLR9l">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Quickstart

Distilabel provides all the tools you need to your scalable and reliable pipelines for synthetic data generation and AI-feedback. Pipelines are used to generate data, evaluate models, manipulate data, or any other general task. They are made up of different components: Steps, Tasks and LLMs, which are chained together in a directed acyclic graph (DAG).

- **Steps**: These are the building blocks of your pipeline. Normal steps are used for basic executions like loading data, applying some transformations, or any other general task.
- **Tasks**: These are steps that rely on LLMs and prompts to perform generative tasks. For example, they can be used to generate data, evaluate models or manipulate data.
- **LLMs**: These are the models that will perform the task. They can be local or remote models, and open-source or commercial models.

Pipelines are designed to be scalable and reliable. They can be executed in a distributed manner, and they can be cached and recovered. This is useful when dealing with large datasets or when you want to ensure that your pipeline is reproducible.

Besides that, pipelines are designed to be modular and flexible. You can easily add new steps, tasks, or LLMs to your pipeline, and you can also easily modify or remove them. An example architecture of a pipeline to generate a dataset of preferences is the following:

## Installation

To install the latest release with `hf-inference-endpoints` extra of the package from PyPI you can use the following command:

```sh
pip install distilabel[hf-inference-endpoints] --upgrade
```

## Define a pipeline

In this guide we will walk you through the process of creating a simple pipeline that uses the [`InferenceEndpointsLLM`][distilabel.models.llms.InferenceEndpointsLLM] class to generate text. The [`Pipeline`][distilabel.pipeline.Pipeline] will load a dataset that contains a column named `prompt` from the Hugging Face Hub via the step [`LoadDataFromHub`][distilabel.steps.LoadDataFromHub] and then use the [`InferenceEndpointsLLM`][distilabel.models.llms.InferenceEndpointsLLM] class to generate text based on the dataset using the [`TextGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/) task.

> You can check the available models in the [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) and filter by `Inference status`.

```python
from distilabel.models import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration

with Pipeline(  # (1)
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:  # (2)
    load_dataset = LoadDataFromHub(  # (3)
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(  # (4)
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),  # (5)
        system_prompt="You are a creative AI Assistant writer.",
        template="Follow the following instruction: {{ instruction }}"  # (6)
    )

    load_dataset >> text_generation  # (7)

if __name__ == "__main__":
    distiset = pipeline.run(  # (8)
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
    distiset.push_to_hub(repo_id="distilabel-example")  # (9)
```

1. We define a [`Pipeline`][distilabel.pipeline.Pipeline] with the name `simple-text-generation-pipeline` and a description `A simple text generation pipeline`. Note that the `name` is mandatory and will be used to calculate the `cache` signature path, so changing the name will change the cache path and will be identified as a different pipeline.

2. We are using the [`Pipeline`][distilabel.pipeline.Pipeline] context manager, meaning that every [`Step`][distilabel.steps.base.Step] subclass that is defined within the context manager will be added to the pipeline automatically.

3. We define a [`LoadDataFromHub`][distilabel.steps.LoadDataFromHub] step named `load_dataset` that will load a dataset from the Hugging Face Hub, as provided via runtime parameters in the `pipeline.run` method below, but it can also be defined within the class instance via the arg `repo_id=...`. This step will produce output batches with the rows from the dataset, and the column `prompt` will be mapped to the `instruction` field.

4. We define a [`TextGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/) task named `text_generation` that will generate text based on the `instruction` field from the dataset. This task will use the [`InferenceEndpointsLLM`][distilabel.models.llms.InferenceEndpointsLLM] class with the model `Meta-Llama-3.1-8B-Instruct`.

5. We define the [`InferenceEndpointsLLM`][distilabel.models.llms.InferenceEndpointsLLM] class with the model `Meta-Llama-3.1-8B-Instruct` that will be used by the [`TextGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/) task. In this case, since the [`InferenceEndpointsLLM`][distilabel.models.llms.InferenceEndpointsLLM] is used, we assume that the `HF_TOKEN` environment variable is set.

6. Both `system_prompt` and `template` are optional fields. The `template` must be informed as a string following the [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/templates/#synopsis) template format, and the fields that appear there ("instruction" in this case, which corresponds to the default) must be informed in the `columns` attribute. The component gallery for [`TextGeneration`](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/) has examples to get you started. 

7. We connect the `load_dataset` step to the `text_generation` task using the `rshift` operator, meaning that the output from the `load_dataset` step will be used as input for the `text_generation` task.

8. We run the pipeline with the parameters for the `load_dataset` and `text_generation` steps. The `load_dataset` step will use the repository `distilabel-internal-testing/instruction-dataset-mini` and the `test` split, and the `text_generation` task will use the `generation_kwargs` with the `temperature` set to `0.7` and the `max_new_tokens` set to `512`.

9. Optionally, we can push the generated [`Distiset`][distilabel.distiset.Distiset] to the Hugging Face Hub repository `distilabel-example`. This will allow you to share the generated dataset with others and use it in other pipelines.
