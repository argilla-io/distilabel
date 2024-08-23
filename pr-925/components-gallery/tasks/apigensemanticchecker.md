---
hide:
  - navigation
---
# APIGenSemanticChecker

Generate queries and answers for the given functions in JSON format.



The `APIGenGenerator` is inspired by the APIGen pipeline, which was designed to generate
    verifiable and diverse function-calling datasets. The task generates a set of diverse queries
    and corresponding answers for the given functions in JSON format.








### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[func_desc]
			ICOL1[query]
			ICOL2[func_call]
			ICOL3[execution_result]
		end
		subgraph New columns
			OCOL0[thought]
			OCOL1[passes]
		end
	end

	subgraph APIGenSemanticChecker
		StepInput[Input Columns: func_desc, query, func_call, execution_result]
		StepOutput[Output Columns: thought, passes]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	ICOL2 --> StepInput
	ICOL3 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepInput --> StepOutput

```


#### Inputs


- **func_desc** (`str`): Description of what the function should do.

- **query** (`str`): Instruction from the user.

- **func_call** (`str`): JSON encoded list with arguments to be passed to the function/API.

- **execution_result** (`str`): Result of the function/API executed.




#### Outputs


- **thought** (`str`): Reasoning for the output in "passes".

- **passes** (`str`): "yes" or "no".







### References

- [APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets](https://arxiv.org/abs/2406.18518)

- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)


