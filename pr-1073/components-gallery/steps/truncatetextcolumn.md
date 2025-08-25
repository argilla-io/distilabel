---
hide:
  - navigation
---
# TruncateTextColumn

Truncate a row using a tokenizer or the number of characters.



`TruncateTextColumn` is a `Step` that truncates a row according to the max length. If
    the `tokenizer` is provided, then the row will be truncated using the tokenizer,
    and the `max_length` will be used as the maximum number of tokens, otherwise it will
    be used as the maximum number of characters. The `TruncateTextColumn` step is useful when one
    wants to truncate a row to a certain length, to avoid posterior errors in the model due
    to the length.





### Attributes

- **column**: the column to truncate. Defaults to `"text"`.

- **max_length**: the maximum length to use for truncation.  If a `tokenizer` is given, corresponds to the number of tokens,  otherwise corresponds to the number of characters. Defaults to `8192`.

- **tokenizer**: the name of the tokenizer to use. If provided, the row will be  truncated using the tokenizer. Defaults to `None`.





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

	subgraph TruncateTextColumn
		StepInput[Input Columns: dynamic]
		StepOutput[Output Columns: dynamic]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **dynamic** (determined by `column` attribute): The columns to be truncated, defaults to "text".




#### Outputs


- **dynamic** (determined by `column` attribute): The truncated column.





### Examples


#### Truncating a row to a given number of tokens
```python
from distilabel.steps import TruncateTextColumn

trunc = TruncateTextColumn(
    tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct",
    max_length=4,
    column="text"
)

trunc.load()

result = next(
    trunc.process(
        [
            {"text": "This is a sample text that is longer than 10 characters"}
        ]
    )
)
# result
# [{'text': 'This is a sample'}]
```

#### Truncating a row to a given number of characters
```python
from distilabel.steps import TruncateTextColumn

trunc = TruncateTextColumn(max_length=10)

trunc.load()

result = next(
    trunc.process(
        [
            {"text": "This is a sample text that is longer than 10 characters"}
        ]
    )
)
# result
# [{'text': 'This is a '}]
```




