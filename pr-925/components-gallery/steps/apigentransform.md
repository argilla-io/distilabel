---
hide:
  - navigation
---
# APIGenTransform

Helper step to transform a dataset like



https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k in into examples
    for the `APIGenGenerator` task.

    Given the rows in formatted as in that dataset, this step prepares the input to be
    passed to the `APIGenGenerator` task by sampling at the batch size.





### Attributes

- **example_template**: String template to format the examples, comes with a default.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[query]
			ICOL1[answers]
			ICOL2[tools]
		end
		subgraph New columns
			OCOL0[examples]
			OCOL1[func_name]
			OCOL2[func_desc]
		end
	end

	subgraph APIGenTransform
		StepInput[Input Columns: query, answers, tools]
		StepOutput[Output Columns: examples, func_name, func_desc]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **query** (`str`): The query that requires an answer in tool format.

- **answers** (`str`): String formatted dict.

- **tools** (`str`): String with formatted list of dictionaries containing the available  tools.




#### Outputs


- **examples** (`str`): Query and answer formatted as an example to be feed  to the prompt.

- **func_name** (`str`): Example name for a function.

- **func_desc** (`str`): Description of the function `func_name`.





### Examples


#### Transform the data for APIGenGenerator
```python
from datasets import load_dataset
from distilabel.steps.tasks.apigen.base import APIGenTransform

samples = load_dataset("Salesforce/xlam-function-calling-60k", split="train").select(range(3)).to_list()
transform = APIGenTransform()
transform.load()
outputs = next(transform.process(samples))
outputs
# [{'examples': '## Query:
What is the T3MA for 'ETH/BTC' using a 1h interval and a time period of 14?
## Answer:
[{"name": "t3ma", "arguments": {"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}}]',
# 'func_name': 'live_giveaways_by_type',
# 'func_desc': 'Retrieve live giveaways from the GamerPower API based on the specified type.'},
# {'examples': '## Query:
Where can I find live giveaways for beta access and games?
## Answer:
[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}]',
# 'func_name': 'web_chain_details',
# 'func_desc': 'python'},
# {'examples': '## Query:
Where can I find live giveaways for beta access and games?
## Answer:
[{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}, {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}]',
# 'func_name': 't3ma',
# 'func_desc': 'Fetches the Triple Exponential Moving Average (T3MA) for a given financial instrument.'}]
```




### References

- [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)

- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)


