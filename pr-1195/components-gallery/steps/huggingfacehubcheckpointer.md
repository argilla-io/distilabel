---
hide:
  - navigation
---
# HuggingFaceHubCheckpointer

Special type of step that uploads the data to a Hugging Face Hub dataset.



A `Step` that uploads the data to a Hugging Face Hub dataset. The data is uploaded in JSONL format
    in a specific Hugging Face Dataset, which can be different to the one where the main distiset
    pipeline is saved. The data is checked every `input_batch_size` inputs, and a new file is created
    in the `repo_id` repository. There will be different config files depending on the leaf steps
    as in the pipeline, and each file will be numbered sequentially. As there will be writes every
    `input_batch_size` inputs, it's advisable not to set a small number on this step, as that
    will slow down the process.





### Attributes

- **repo_id**: The ID of the repository to push to in the following format: `<user>/<dataset_name>` or  `<org>/<dataset_name>`. Also accepts `<dataset_name>`, which will default to the namespace  of the logged-in user.

- **private**: Whether the dataset repository should be set to private or not. Only affects repository creation:  a repository that already exists will not be affected by that parameter.

- **token**: An optional authentication token for the Hugging Face Hub. If no token is passed, will default  to the token saved locally when logging in with `huggingface-cli login`. Will raise an error  if no token is passed and the user is not logged-in.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
	end

	subgraph HuggingFaceHubCheckpointer
	end


```







### Examples


#### Do checkpoints of the data generated in a Hugging Face Hub dataset
```python
from typing import TYPE_CHECKING
from datasets import Dataset

from distilabel.pipeline import Pipeline
from distilabel.steps import HuggingFaceHubCheckpointer
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.typing import StepOutput

# Create a dummy dataset
dataset = Dataset.from_dict({"instruction": ["tell me lies"] * 100})

with Pipeline(name="pipeline-with-checkpoints") as pipeline:
    text_generation = TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
        template="Follow the following instruction: {{ instruction }}"
    )
    checkpoint = HuggingFaceHubCheckpointer(
        repo_id="username/streaming_checkpoint",
        private=True,
        input_batch_size=50  # Will save write the data to the dataset every 50 inputs
    )
    text_generation >> checkpoint
```




