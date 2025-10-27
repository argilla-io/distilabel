---
hide:
  - navigation
---
# TextGenerationToArgilla

Creates a text generation dataset in Argilla.



`Step` that creates a dataset in Argilla during the load phase, and then pushes the input
    batches into it as records. This dataset is a text-generation dataset, where there's one field
    per each input, and then a label question to rate the quality of the completion in either bad
    (represented with 👎) or good (represented with 👍).



### Note
This step is meant to be used in conjunction with a `TextGeneration` step and no column mapping
is needed, as it will use the default values for the `instruction` and `generation` columns.



### Attributes

- **dataset_name**: The name of the dataset in Argilla.

- **dataset_workspace**: The workspace where the dataset will be created in Argilla. Defaults to  `None`, which means it will be created in the default workspace.

- **api_url**: The URL of the Argilla API. Defaults to `None`, which means it will be read from  the `ARGILLA_API_URL` environment variable.

- **api_key**: The API key to authenticate with Argilla. Defaults to `None`, which means it will  be read from the `ARGILLA_API_KEY` environment variable.




### Runtime Parameters

- **api_url**: The base URL to use for the Argilla API requests.

- **api_key**: The API key to authenticate the requests to the Argilla API.



### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instruction]
			ICOL1[generation]
		end
	end

	subgraph TextGenerationToArgilla
		StepInput[Input Columns: instruction, generation]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput

```


#### Inputs


- **instruction** (`str`): The instruction that was used to generate the completion.

- **generation** (`str` or `List[str]`): The completions that were generated based on the input instruction.







### Examples


#### Push a text generation dataset to an Argilla instance
```python
from distilabel.steps import PreferenceToArgilla

to_argilla = TextGenerationToArgilla(
    num_generations=2,
    api_url="https://dibt-demo-argilla-space.hf.space/",
    api_key="api.key",
    dataset_name="argilla_dataset",
    dataset_workspace="my_workspace",
)
to_argilla.load()

result = next(
    to_argilla.process(
        [
            {
                "instruction": "instruction",
                "generation": "generation",
            }
        ],
    )
)
# >>> result
# [{'instruction': 'instruction', 'generation': 'generation'}]
```




