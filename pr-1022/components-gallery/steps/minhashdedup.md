---
hide:
  - navigation
---
# MinHashDedup

Deduplicates text using `MinHash` and `MinHashLSH`.



`MinHashDedup` is a Step that detects near-duplicates in datasets. The idea roughly translates
    to the following steps:
    1. Tokenize the text into words or ngrams.
    2. Create a `MinHash` for each text.
    3. Store the `MinHashes` in a `MinHashLSH`.
    4. Check if the `MinHash` is already in the `LSH`, if so, it is a duplicate.





### Attributes

- **num_perm**: the number of permutations to use. Defaults to `128`.

- **seed**: the seed to use for the MinHash. This seed must be the same  used for `MinHash`, keep in mind when both steps are created. Defaults to `1`.

- **tokenizer**: the tokenizer to use. Available ones are `words` or `ngrams`.  If `words` is selected, it tokenize the text into words using nltk's  word tokenizer. `ngram` estimates the ngrams (together with the size  `n`) using. Defaults to `words`.

- **n**: the size of the ngrams to use. Only relevant if `tokenizer="ngrams"`. Defaults to `5`.

- **threshold**: the threshold to consider two MinHashes as duplicates.  Values closer to 0 detect more duplicates. Defaults to `0.9`.

- **storage**: the storage to use for the LSH. Can be `dict` to store the index  in memory, or `disk`. Keep in mind, `disk` is an experimental feature  not defined in `datasketch`, that is based on DiskCache's `Index` class.  It should work as a `dict`, but backed by disk, but depending on the system  it can be slower. Defaults to `dict`.  which uses a custom `shelve` backend. Note the `disk`  is an experimetal feature that may cause issues. Defaults to `dict`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
		end
		subgraph New columns
			OCOL0[keep_row_after_minhash_filtering]
		end
	end

	subgraph MinHashDedup
		StepInput[Input Columns: text]
		StepOutput[Output Columns: keep_row_after_minhash_filtering]
	end

	ICOL0 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`): the texts to be filtered.




#### Outputs


- **keep_row_after_minhash_filtering** (`bool`): boolean indicating if the piece `text` is  not a duplicate i.e. this text should be kept.





### Examples


#### Deduplicate a list of texts using MinHash and MinHashLSH
```python
from distilabel.pipeline import Pipeline
from distilabel.steps import MinHashDedup
from distilabel.steps import LoadDataFromDicts

with Pipeline() as pipeline:
    ds_size = 1000
    batch_size = 500  # Bigger batch sizes work better for this step
    data = LoadDataFromDicts(
        data=[
            {"text": "This is a test document."},
            {"text": "This document is a test."},
            {"text": "Test document for duplication."},
            {"text": "Document for duplication test."},
            {"text": "This is another unique document."},
        ]
        * (ds_size // 5),
        batch_size=batch_size,
    )
    minhash_dedup = MinHashDedup(
        tokenizer="words",
        threshold=0.9,      # lower values will increase the number of duplicates
        storage="dict",     # or "disk" for bigger datasets
    )

    data >> minhash_dedup

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    ds = distiset["default"]["train"]
    # Filter out the duplicates
    ds_dedup = ds.filter(lambda x: x["keep_row_after_minhash_filtering"])
```




### References

- [datasketch documentation](https://ekzhu.github.io/datasketch/lsh.html)

- [Identifying and Filtering Near-Duplicate Documents](https://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf)

- [Diskcache's Index](https://grantjenks.com/docs/diskcache/api.html#diskcache.Index)


