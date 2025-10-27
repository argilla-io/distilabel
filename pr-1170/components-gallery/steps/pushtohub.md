---
hide:
  - navigation
---
# PushToHub

Push data to a Hugging Face Hub dataset.



A `GlobalStep` which creates a `datasets.Dataset` with the input data and pushes
    it to the Hugging Face Hub.





### Attributes

- **repo_id**: The Hugging Face Hub repository ID where the dataset will be uploaded.

- **split**: The split of the dataset that will be pushed. Defaults to `"train"`.

- **private**: Whether the dataset to be pushed should be private or not. Defaults to  `False`.

- **token**: The token that will be used to authenticate in the Hub. If not provided, the  token will be tried to be obtained from the environment variable `HF_TOKEN`.  If not provided using one of the previous methods, then `huggingface_hub` library  will try to use the token from the local Hugging Face CLI configuration. Defaults  to `None`.




### Runtime Parameters

- **repo_id**: The Hugging Face Hub repository ID where the dataset will be uploaded.

- **split**: The split of the dataset that will be pushed.

- **private**: Whether the dataset to be pushed should be private or not.

- **token**: The token that will be used to authenticate in the Hub.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[dynamic]
		end
	end

	subgraph PushToHub
		StepInput[Input Columns: dynamic]
	end

	ICOL0 --> StepInput

```


#### Inputs


- **dynamic** (`all`): all columns from the input will be used to create the dataset.







### Examples


#### Push batches of your dataset to the Hugging Face Hub repository
```python
from distilabel.steps import PushToHub

push = PushToHub(repo_id="path_to/repo")
push.load()

result = next(
    push.process(
        [
            {
                "instruction": "instruction ",
                "generation": "generation"
            }
        ],
    )
)
# >>> result
# [{'instruction': 'instruction ', 'generation': 'generation'}]
```




