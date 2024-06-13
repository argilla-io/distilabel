---
hide:
  - navigation
---
# PreferenceToArgilla

Creates a preference dataset in Argilla.



Step that creates a dataset in Argilla during the load phase, and then pushes the input
    batches into it as records. This dataset is a preference dataset, where there's one field
    for the instruction and one extra field per each generation within the same record, and then
    a rating question per each of the generation fields. The rating question asks the annotator to
    set a rating from 1 to 5 for each of the provided generations.



### Note
This step is meant to be used in conjunction with the `UltraFeedback` step, or any other step
generating both ratings and responses for a given set of instruction and generations for the
given instruction. But alternatively, it can also be used with any other task or step generating
only the `instruction` and `generations`, as the `ratings` and `rationales` are optional.



### Attributes

- **num_generations**: The number of generations to include in the dataset.

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
			ICOL1[generations]
			ICOL2[ratings]
			ICOL3[rationales]
		end
	end

	subgraph PreferenceToArgilla
		StepInput[Input Columns: instruction, generations, ratings, rationales]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	ICOL3 --> StepInput

```


#### Inputs


- **instruction** (`str`): The instruction that was used to generate the completion.

- **generations** (`List[str]`): The completion that was generated based on the input instruction.

- **ratings** (`List[str]`, optional): The ratings for the generations. If not provided, the  generated ratings won't be pushed to Argilla.

- **rationales** (`List[str]`, optional): The rationales for the ratings. If not provided, the  generated rationales won't be pushed to Argilla.







### Examples


#### Push a preference dataset to an Argilla instance
```python
from distilabel.steps import PreferenceToArgilla

to_argilla = PreferenceToArgilla(
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
                "generations": ["first_generation", "second_generation"],
            }
        ],
    )
)
# >>> result
# [{'instruction': 'instruction', 'generations': ['first_generation', 'second_generation']}]
```

#### It can also include ratings and rationales
```python
result = next(
    to_argilla.process(
        [
            {
                "instruction": "instruction",
                "generations": ["first_generation", "second_generation"],
                "ratings": ["4", "5"],
                "rationales": ["rationale for 4", "rationale for 5"],
            }
        ],
    )
)
# >>> result
# [
#     {
#         'instruction': 'instruction',
#         'generations': ['first_generation', 'second_generation'],
#         'ratings': ['4', '5'],
#         'rationales': ['rationale for 4', 'rationale for 5']
#     }
# ]
```




