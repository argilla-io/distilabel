---
hide:
  - navigation
---
# MergeColumns

Merge columns from a row.



`MergeColumns` is a `Step` that implements the `process` method that calls the `merge_columns`
    function to handle and combine columns in a `StepInput`. `MergeColumns` provides two attributes
    `columns` and `output_column` to specify the columns to merge and the resulting output column.

    This step can be useful if you have a `Task` that generates instructions for example, and you
    want to have more examples of those. In such a case, you could for example use another `Task`
    to multiply your instructions synthetically, what would yield two different columns splitted.
    Using `MergeColumns` you can merge them and use them as a single column in your dataset for
    further processing.





### Attributes

- **columns**: List of strings with the names of the columns to merge.

- **output_column**: str name of the output column





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

	subgraph MergeColumns
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `columns` attribute): The columns to merge.




#### Outputs


- **dynamic** (determined by `columns` and `output_column` attributes): The columns  that were merged.





### Examples


#### Combine columns in rows of a dataset
```python
from distilabel.steps import MergeColumns

combiner = MergeColumns(
    columns=["queries", "multiple_queries"],
    output_column="queries",
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




