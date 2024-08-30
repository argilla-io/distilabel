---
hide:
  - navigation
---
# MinHash

Creates the components for a `MinHash` object to deduplicate texts.



From `datasketch` documentation:
    Estimates the Jaccard similarity (resemblance) between sets of arbitrary sizes in linear
    time using a small and fixed memory space.



### Note
We only keep the hashvalues, as using those values together with the seed
we can reproduce the `MinHash` objects. The `MinHashLSH` will recreate those internally.



### Attributes

- **num_perm**: the number of permutations to use. Defaults to `128`.

- **seed**: the seed to use for the MinHash. Defaults to `1`.

- **tokenizer**: the tokenizer to use. Available ones are `words` or `ngrams`.  If `words` is selected, it tokenize the text into words using nltk's  word tokenizer. `ngram` estimates the ngrams (together with the size  `n`) using. Defaults to `words`.

- **n**: the size of the ngrams to use. Only relevant if `tokenizer="ngrams"`. Defaults to `1`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
		end
		subgraph New columns
			OCOL0[hashvalues]
		end
	end

	subgraph MinHash
		StepInput[Input Columns: text]
		StepOutput[Output Columns: hashvalues]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`): the texts to obtain the hashes for.




#### Outputs


- **hashvalues** (`List[int]`): hash values obtained for the algorithm.





### Examples


#### Create MinHash objects for a list of texts to be deduplicated
```python
texts: List[str] = [
    "This is a test document.",
    "This document is a test.",
    "Test document for duplication.",
    "Document for duplication test.",
    "This is another unique document."
]
from distilabel.steps import MinHash
minhasher = MinHash(tokenizer="ngrams", n=3)
minhasher.load()
result = next(hasher.process([{"text": t} for t in texts]))
```




### References

- [datasketch documentation](https://ekzhu.com/datasketch/minhash.html#minhash)

- [Identifying and Filtering Near-Duplicate Documents](https://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf)


