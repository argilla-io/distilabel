---
hide:
  - navigation
---
# MinHashLSH

Creates a `MinHashLSH` index to deduplicate texts using MinHash.



This class must be used together with `MinHash` step. It will work with the previous hashes
    to detect duplicate texts, and inform whether a given row can be removed.





### Attributes

- **seed**: the seed to use for the MinHash. This seed must be the same  used for `MinHash`, keep in mind when both steps are created. Defaults to `1`.

- **num_perm**: the number of permutations to use. Defaults to `128`.

- **threshold**: the threshold to consider two MinHashes as duplicates.  Values closer to 0 detect more duplicates. Defaults to `0.9`.

- **drop_hashvalues**: whether to drop the hashvalues after processing. Defaults to `False`.

- **storage**: the storage to use for the LSH. Can be `dict` to store the index  in memory, or `disk`, which uses a custom `shelve` backend. Note the `disk`  is an experimetal feature that may cause issues. Defaults to `dict`.





### Input & Output Columns

``` mermaid
graph TD
	subgraph Dataset
		subgraph Columns
			ICOL0[text]
			ICOL1[hashvalues]
		end
		subgraph New columns
			OCOL0[minhash_duplicate]
		end
	end

	subgraph MinHashLSH
		StepInput[Input Columns: text, hashvalues]
		StepOutput[Output Columns: minhash_duplicate]
	end

	ICOL0 --> StepInput
	ICOL1 --> StepInput
	StepOutput --> OCOL0
	StepInput --> StepOutput

```


#### Inputs


- **text** (`str`): the texts to be filtered.

- **hashvalues** (`List[int]`): hash values obtained from `MinHash` step.




#### Outputs


- **minhash_duplicate** (`bool`): boolean indicating if the piece of text is a  duplicate or not, so the user can decide afterwards whether to remove it  or not.





### Examples


#### Deduplicate a list of texts using MinHash and MinHashLSH
```python
from distilabel.pipeline import Pipeline
from distilabel.steps import MinHash, MinHashLSH

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
    minhash = MinHash(tokenizer="ngrams", n=1, input_batch_size=batch_size)
    minhash_lsh = MinHashLSH(
        threshold=0.9,         # lower values will increase the number of duplicates
        seed=minhash.seed,     # we need to keep the same seed for the LSH
        drop_hashvalues=True,  # the hashvalues are not needed anymore
        storage="dict",        # or "disk" for bigger datasets
    )
    data >> minhash >> minhash_lsh

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    ds = distiset["default"]["train"]
    # Filter out the duplicates
    ds_dedup = ds.filter(lambda x: x["minhash_duplicate"] is False)
```




### References

- [datasketch documentation](https://ekzhu.github.io/datasketch/lsh.html)


