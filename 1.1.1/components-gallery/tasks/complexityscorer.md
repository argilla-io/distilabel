# ComplexityScorer


Score instructions based on their complexity using an `LLM`.



`ComplexityScorer` is a pre-defined task used to rank a list of instructions based in
    their complexity. It's an implementation of the complexity score task from the paper
    'What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection
    in Instruction Tuning'.





### Attributes

- **_template**: a Jinja2 template used to format the input for the LLM.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[instructions]
		end
		subgraph New columns
			OCOL0[scores]
			OCOL1[model_name]
		end
	end

	subgraph ComplexityScorer
		StepInput[Input Columns: instructions]
		StepOutput[Output Columns: scores, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **instructions** (`List[str]`): The list of instructions to be scored.




#### Outputs


- **scores** (`List[float]`): The score for each instruction.

- **model_name** (`str`): The model name used to generate the scores.







### References

- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://arxiv.org/abs/2312.15685)


