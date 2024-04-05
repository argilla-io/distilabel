# Global Steps

The global steps are the ones that in order to do it's processing, they will need access to all the data at once. Some examples include creating a dataset to be pushed to the hub, or a filtering step in a `Pipeline`.

## Push data to HuggingFace Hub in batches

The first example of a `global` step corresponds to [`PushToHub`][distilabel.steps.globals.huggingface]:

```python
import os

from distilabel.pipeline.local import Pipeline
from distilabel.steps.globals.huggingface import PushToHub

push_to_hub = PushToHub(
    name="push_to_hub",
    repo_id="org/dataset-name",
    split="train",
    private=False,
    token=os.getenv("HF_API_TOKEN"),
    pipeline=Pipeline(name="push-pipeline"),
)
```

This step can be used to push batches of the dataset to the hub as the process advances, enabling a checkpoint strategy in your pipeline.

## Data Filtering

For some pipelines we may need to filter data according to some criteria. For example, the implementation of [`DeitaFiltering`][distilabel.steps.deita] does some filtering to determine the examples to keep according to ensure the final dataset has enough diversity. We will see this step in it's own place because it may be difficult to follow out of context.