---
hide:
  - navigation
---
# CombineKeys

Combines keys from a row.



`CombineKeys` is a `Step` that implements the `process` method that calls the `combine_keys`
    function to handle and combine keys in a `StepInput`. `CombineKeys` provides two attributes
    `keys` and `output_keys` to specify the keys to merge and the resulting output key.

    This step can be useful if you have a `Task` that generates instructions for example, and you
    want to have more examples of those. In such a case, you could for example use another `Task`
    to multiply your instructions synthetically, what would yield two different keys splitted.
    Using `CombineKeys` you can merge them and use them as a single column in your dataset for
    further processing.





### Attributes

- **columns**: List of strings with the names of the columns to merge.

- **output_columns**: Optional list of strings with the names of the output columns.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[dynamic]
		end
		subgraph New columns
			OCOL0[dynamic]
		end
	end

	subgraph CombineKeys
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `keys` attribute): The keys to merge.




#### Outputs


- **dynamic** (determined by `keys` and `output_key` attributes): The columns  that were merged.





### Examples


#### Combine keys in rows of a dataset
```python
from distilabel.steps import CombineKeys

combiner = CombineKeys(
    keys=["queries", "multiple_queries"],
    output_key="queries",
)
combiner.load()

result = next(
    combiner.process(
        [
            {
                "queries": "How are you?",
                "multiple_queries": ["What's up?", "Everything ok?"]
            }
        ],
    )
)
# >>> result
# [{'queries': ['How are you?', "What's up?", 'Everything ok?']}]
```




