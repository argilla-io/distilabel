# PairRM


Rank the candidates based on the input using the `LLM` model.





### Attributes

- **model**: The model to use for the ranking. Defaults to `"llm-blender/PairRM"`.

- **instructions**: The instructions to use for the model. Defaults to `None`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[inputs]
			ICOL1[candidates]
		end
		subgraph New columns
			OCOL0[ranks]
			OCOL1[ranked_candidates]
			OCOL2[model_name]
		end
	end

	subgraph PairRM
		StepInput[Input Columns: inputs, candidates]
		StepOutput[Output Columns: ranks, ranked_candidates, model_name]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **inputs** (`List[Dict[str, Any]]`): The input text or conversation to rank the candidates for.

- **candidates** (`List[Dict[str, Any]]`): The candidates to rank.




#### Outputs


- **ranks** (`List[int]`): The ranks of the candidates based on the input.

- **ranked_candidates** (`List[Dict[str, Any]]`): The candidates ranked based on the input.

- **model_name** (`str`): The model name used to rank the candidate responses. Defaults to `"llm-blender/PairRM"`.







### References

- [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561)

- [Pair Ranking Model](https://huggingface.co/llm-blender/PairRM)


