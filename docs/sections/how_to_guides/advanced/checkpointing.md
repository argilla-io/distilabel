# Push data to the hub while the pipeline is running

Long-running pipelines can be resource-intensive, and ensuring everything is functioning as expected is crucial. To make this process seamless, the [`HuggingFaceHubCheckpointer`][distilabel.steps.HuggingFaceHubCheckpointer] step has been designed to integrate directly into the pipeline workflow.

The [`HuggingFaceHubCheckpointer`][distilabel.steps.HuggingFaceHubCheckpointer] allows you to periodically save your generated data as a Hugging Face Dataset at configurable intervals (every `input_batch_size` examples generated).

Just add the [`HuggingFaceHubCheckpointer`][distilabel.steps.HuggingFaceHubCheckpointer] as any other step in your pipeline.

## Sample pipeline with dummy data to see the checkpoint strategy in action

The following pipeline starts from a fake dataset with dummy data, passes that through a fake `DoNothing` step (any other step/s work here, but this can be useful to explore the behavior), and makes use of the [`HuggingFaceHubCheckpointer`][distilabel.steps.HuggingFaceHubCheckpointer] step to push the data to the hub.

```python
from datasets import Dataset

from distilabel.pipeline import Pipeline
from distilabel.steps import HuggingFaceHubCheckpointer
from distilabel.steps.base import Step, StepInput
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

dataset = Dataset.from_dict({"a": [1, 2, 3, 4] * 50, "b": [5, 6, 7, 8] * 50})

class DoNothing(Step):
    def process(self, *inputs: StepInput) -> "StepOutput":
        for input in inputs:
            yield input

with Pipeline(name="pipeline-with-checkpoints") as pipeline:
    text_generation = DoNothing(
        input_batch_size=60
    )
    checkpoint = HuggingFaceHubCheckpointer(
        repo_id="username/streaming_test_1",  # (1)
        private=True,
        input_batch_size=50  # (2)
    )
    text_generation >> checkpoint


if __name__ == "__main__":
    distiset = pipeline.run(
        dataset=dataset,
        use_cache=False
    )
    distiset.push_to_hub(repo_id="username/streaming_test")
```

1. The name of the dataset for the checkpoints, can be different to the final distiset. This dataset
will contain less information than the final distiset to make it faster while the pipeline is running.
2. The `input_batch_size` determines how often the data is pushed to the Hugging Face Hub. If the process is really slow, say for a big model, a value like 100 may be on point, for smaller models or pipelines that generate data faster, 10.000 maybe more relevant. It's better to explore the value for a given use case.

The final datasets can be found in the following links:

- Checkpoint dataset: [distilabel-internal-testing/streaming_test_1](https://huggingface.co/datasets/distilabel-internal-testing/streaming_test_1)

- Final distiset: [distilabel-internal-testing/streaming_test](https://huggingface.co/datasets/distilabel-internal-testing/streaming_test)
