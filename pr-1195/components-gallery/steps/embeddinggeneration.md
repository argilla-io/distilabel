---
hide:
  - navigation
---
# EmbeddingGeneration

Generate embeddings using an `Embeddings` model.



`EmbeddingGeneration` is a `Step` that using an `Embeddings` model generates sentence
    embeddings for the provided input texts.





### Attributes

- **embeddings**: the `Embeddings` model used to generate the sentence embeddings.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
		end
		subgraph New columns
			OCOL0[embedding]
		end
	end

	subgraph EmbeddingGeneration
		StepInput[Input Columns: text]
		StepOutput[Output Columns: embedding]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`): The text for which the sentence embedding has to be generated.




#### Outputs


- **embedding** (`List[Union[float, int]]`): the generated sentence embedding.





### Examples


#### Generate sentence embeddings with Sentence Transformers
```python
from distilabel.models import SentenceTransformerEmbeddings
from distilabel.steps import EmbeddingGeneration

embedding_generation = EmbeddingGeneration(
    embeddings=SentenceTransformerEmbeddings(
        model="mixedbread-ai/mxbai-embed-large-v1",
    )
)

embedding_generation.load()

result = next(embedding_generation.process([{"text": "Hello, how are you?"}]))
# [{'text': 'Hello, how are you?', 'embedding': [0.06209656596183777, -0.015797119587659836, ...]}]
```




