---
hide:
  - navigation
---
# GenerateSentencePair

Generate a positive and negative (optionally) sentences given an anchor sentence.



`GenerateSentencePair` is a pre-defined task that given an anchor sentence generates
    a positive sentence related to the anchor and optionally a negative sentence unrelated
    to the anchor or similar to it. Optionally, you can give a context to guide the LLM
    towards more specific behavior. This task is useful to generate training datasets for
    training embeddings models.





### Attributes

- **triplet**: a flag to indicate if the task should generate a triplet of sentences  (anchor, positive, negative). Defaults to `False`.

- **action**: the action to perform to generate the positive sentence.

- **context**: the context to use for the generation. Can be helpful to guide the LLM  towards more specific context. Not used by default.

- **hard_negative**: A flag to indicate if the negative should be a hard-negative or not.  Hard negatives make it hard for the model to distinguish against the positive,  with a higher degree of semantic similarity.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[anchor]
		end
		subgraph New columns
			OCOL0[positive]
			OCOL1[negative]
			OCOL2[model_name]
		end
	end

	subgraph GenerateSentencePair
		StepInput[Input Columns: anchor]
		StepOutput[Output Columns: positive, negative, model_name]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepOutput --> OCOL1
	StepOutput --> OCOL2
	StepInput --> StepOutput

```


#### Inputs


- **anchor** (`str`): The anchor sentence to generate the positive and negative sentences.




#### Outputs


- **positive** (`str`): The positive sentence related to the `anchor`.

- **negative** (`str`): The negative sentence unrelated to the `anchor` if `triplet=True`,  or more similar to the positive to make it more challenging for a model to distinguish  in case `hard_negative=True`.

- **model_name** (`str`): The name of the model that was used to generate the sentences.





### Examples


#### Paraphrasing
```python
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.models import InferenceEndpointsLLM

generate_sentence_pair = GenerateSentencePair(
    triplet=True, # `False` to generate only positive
    action="paraphrase",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    input_batch_size=10,
)

generate_sentence_pair.load()

result = generate_sentence_pair.process([{"anchor": "What Game of Thrones villain would be the most likely to give you mercy?"}])
```

#### Generating semantically similar sentences
```python
from distilabel.models import InferenceEndpointsLLM
from distilabel.steps.tasks import GenerateSentencePair

generate_sentence_pair = GenerateSentencePair(
    triplet=True, # `False` to generate only positive
    action="semantically-similar",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    input_batch_size=10,
)

generate_sentence_pair.load()

result = generate_sentence_pair.process([{"anchor": "How does 3D printing work?"}])
```

#### Generating queries
```python
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.models import InferenceEndpointsLLM

generate_sentence_pair = GenerateSentencePair(
    triplet=True, # `False` to generate only positive
    action="query",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    input_batch_size=10,
)

generate_sentence_pair.load()

result = generate_sentence_pair.process([{"anchor": "Argilla is an open-source data curation platform for LLMs. Using Argilla, ..."}])
```

#### Generating answers
```python
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.models import InferenceEndpointsLLM

generate_sentence_pair = GenerateSentencePair(
    triplet=True, # `False` to generate only positive
    action="answer",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    input_batch_size=10,
)

generate_sentence_pair.load()

result = generate_sentence_pair.process([{"anchor": "What Game of Thrones villain would be the most likely to give you mercy?"}])
```

#### )
```python
from distilabel.steps.tasks import GenerateSentencePair
from distilabel.models import InferenceEndpointsLLM

generate_sentence_pair = GenerateSentencePair(
    triplet=True, # `False` to generate only positive
    action="query",
    context="Argilla is an open-source data curation platform for LLMs.",
    hard_negative=True,
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    input_batch_size=10,
    use_default_structured_output=True
)

generate_sentence_pair.load()

result = generate_sentence_pair.process([{"anchor": "I want to generate queries for my LLM."}])
```




