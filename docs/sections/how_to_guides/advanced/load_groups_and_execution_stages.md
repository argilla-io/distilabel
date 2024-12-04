# Load groups and execution stages

By default, the `distilabel` architecture loads all steps of a pipeline at the same time, as they are all supposed to process batches of data in parallel. However, loading all steps at once can waste resources in two scenarios: when using `GlobalStep`s that must wait for upstream steps to complete before processing data, or when running on machines with limited resources that cannot execute all steps simultaneously. In these cases, steps need to be loaded and executed in distinct **load stages**.

## Load stages

A load stage represents a point in the pipeline execution where a group of steps are loaded at the same time to process batches in parallel. These stages are required because:

1. There are some kind of steps like the `GlobalStep`s that needs to receive all the data at once from their upstream steps i.e. needs their upstream steps to have finished its execution. It would be wasteful to load a `GlobalStep` at the same time as other steps of the pipeline as that would take resources (from the machine or cluster running the pipeline) that wouldn't be used until upstream steps have finished.
2. When running on machines or clusters with limited resources, it may be not possible to load and execute all steps simultaneously as they would need to access the same limited resources (memory, CPU, GPU, etc.). 

Having that said, the first element that will create a load stage when executing a pipeline are the [`GlobalStep`][distilabel.steps.base.GlobalStep], as they mark and divide a pipeline in three stages: one stage with the upstream steps of the global step, one stage with the global step, and one final stage with the downstream steps of the global step. For example, the following pipeline will contain three stages:

```python
from typing import TYPE_CHECKING

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, StepInput, step

if TYPE_CHECKING:
    from distilabel.typing import StepOutput


@step(inputs=["instruction"], outputs=["instruction2"])
def DummyStep(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction2"] = "miau"
    yield inputs


@step(inputs=["instruction"], outputs=["instruction2"], step_type="global")
def GlobalDummyStep(inputs: StepInput) -> "StepOutput":
    for input in inputs:
        input["instruction2"] = "miau"
    yield inputs


with Pipeline() as pipeline:
    generator = LoadDataFromDicts(data=[{"instruction": "Hi"}] * 50)
    dummy_step_0 = DummyStep()
    global_dummy_step = GlobalDummyStep()
    dummy_step_1 = DummyStep()

    generator >> dummy_step_0 >> global_dummy_step >> dummy_step_1

if __name__ == "__main__":
    load_stages = pipeline.get_load_stages()

    for i, steps_stage in enumerate(load_stages[0]):
        print(f"Stage {i}: {steps_stage}")

    # Output:
    # Stage 0: ['load_data_from_dicts_0', 'dummy_step_0']
    # Stage 1: ['global_dummy_step_0']
    # Stage 2: ['dummy_step_1']
```

As we can see, the `GlobalStep` divided the pipeline execution in three stages.

## Load groups

While `GlobalStep`s automatically divide pipeline execution into stages, we many need fine-grained control over how steps are loaded and executed within each stage. This is where **load groups** come in.

Load groups allows to specify which steps of the pipeline have to be loaded together within a stage. This is particularly useful when running on resource-constrained environments where all the steps cannot be executed in parallel.

Let's see how it works with an example:

```python
from datasets import load_dataset

from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration

dataset = load_dataset(
    "distilabel-internal-testing/instruction-dataset-mini", split="test"
).rename_column("prompt", "instruction")

with Pipeline() as pipeline:
    text_generation_0 = TextGeneration(
        llm=vLLM(
            model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            extra_kwargs={"max_model_len": 1024},
        ),
        resources=StepResources(gpus=1),
    )

    text_generation_1 = TextGeneration(
        llm=vLLM(
            model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            extra_kwargs={"max_model_len": 1024},
        ),
        resources=StepResources(gpus=1),
    )

if __name__ == "__main__":
    load_stages = pipeline.get_load_stages(load_groups=[[text_generation_1.name]])

    for i, steps_stage in enumerate(load_stages[0]):
        print(f"Stage {i}: {steps_stage}")

    # Output:
    # Stage 0: ['text_generation_0']
    # Stage 1: ['text_generation_1']

    distiset = pipeline.run(dataset=dataset, load_groups=[[text_generation_0.name]])
```

In this example, we're working with a machine that has a single GPU, but the pipeline includes two instances of [TextGeneration]() tasks both using [vLLM]() and requesting 1 GPU. We cannot execute both steps in parallel. To fix that,
we specify in the `run` method using the `load_groups` argument that the `text_generation_0` step has to be executed in isolation in a stage. This way, we can run the pipeline on a single GPU machine by executing the steps in different stages (sequentially) instead of in parallel.

Some key points about load groups:

1. Load groups are specified as a list of lists, where each inner list represents a group of steps that should be loaded together.
2. Same as `GlobalSteps`s, the load groups creates a new load stage dividing the pipeline in 3 stages: one for the upstream steps, one for the steps in the load group, and one for the downstream steps.
