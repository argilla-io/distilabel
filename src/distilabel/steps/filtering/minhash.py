# Copyright 2023-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import uuid
from functools import partial
from itertools import tee
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.base import GlobalStep, Step, StepInput

if TYPE_CHECKING:
    from datasketch import LeanMinHash, MinHash, MinHashLSH

    from distilabel.steps.typing import StepOutput


# Copied from: https://github.com/huggingface/datatrove/blob/main/src/datatrove/utils/text.py#L89C1-L95C65
def ngrams(sequence: Iterable[str], n: int) -> Iterator[Tuple[str, ...]]:
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


def tokenized_on_words(texts: Iterable[str]) -> List[Set[bytes]]:
    """Tokenizes a list of texts into words, using `nltk.word_tokenize`.

    Args:
        texts: List of documents to be tokenized.

    Returns:
        List with the set of tokens for each document.
    """
    from nltk.tokenize import word_tokenize

    return [{w.encode("utf-8") for w in word_tokenize(text)} for text in texts]


def tokenize_on_ngrams(texts: Iterable[str], n: int = 1) -> List[Set[bytes]]:
    """Tokenizes a list of texts into ngrams, and returns the set of them as bytes.

    Args:
        texts: List of documents to be tokenized.
        n: The size of the ngrams, defaults to 1 (single letters).

    Returns:
        List with the set of tokens for each document.
    """

    return [
        {"".join(ngram).encode("utf-8") for ngram in ngrams(text, n=n)}
        for text in texts
    ]


# NOTE: This class must be used together with the `MinHashLSH` class.
# We return the `hashvalues` to reproduce the MinHash objects, but we also need
# the seed, so the seed used for the MinHash objects must be kept to be grabbed
# for the next class. We could also pass it as part of the dict, but there's no point.
# Also, instead of returning the values, we could be saving them as artifacts,
# This still needs to be studied.
class MinHash(Step):
    """Creates the components for a `MinHash` object to deduplicate texts.

    From `datasketch` documentation:
    Estimates the Jaccard similarity (resemblance) between sets of arbitrary sizes in linear
    time using a small and fixed memory space.

    Note:
        We only keep the hashvalues, as using those values together with the seed
        we can reproduce the `MinHash` objects. The `MinHashLSH` will recreate those internally.

    Attributes:
        num_perm: the number of permutations to use. Defaults to `128`.
        seed: the seed to use for the MinHash. Defaults to `1`.
        tokenizer: the tokenizer to use. Available ones are `words` or `ngrams`.
            If `words` is selected, it tokenize the text into words using nltk's
            word tokenizer. `ngram` estimates the ngrams (together with the size
            `n`) using. Defaults to `words`.
        n: the size of the ngrams to use. Only relevant if `tokenizer="ngrams"`. Defaults to `1`.

    Input columns:
        - text (`str`): the texts to obtain the hashes for.

    Output columns:
        - hashvalues (`List[int]`): hash values obtained for the algorithm.

    Categories:
        - filtering

    References:
        - [`datasketch documentation`](https://ekzhu.com/datasketch/minhash.html#minhash)
        - [Identifying and Filtering Near-Duplicate Documents](https://cs.brown.edu/courses/cs253/papers/nearduplicate.pdf)

    Examples:

        Create MinHash objects for a list of texts to be deduplicated:

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
    """

    num_perm: int = 128
    seed: int = 1
    tokenizer: Literal["words", "ngrams"] = "words"
    n: Optional[int] = 1
    _hasher: Union["MinHash", None] = PrivateAttr(None)
    _tokenizer: Union[Callable, None] = PrivateAttr(None)

    def load(self) -> None:
        super().load()
        if not importlib.import_module("datasketch"):
            raise ImportError(
                "`datasketch` is needed to deduplicate with MinHash, but is not installed. "
                "Please install it using `pip install datasketch`."
            )
        from datasketch import MinHash

        self._hasher = MinHash.bulk

        if self.tokenizer == "words":
            if not importlib.import_module("nltk"):
                raise ImportError(
                    "`nltk` is needed to tokenize based on words, but is not installed. "
                    "Please install it using `pip install nltk`. Then run `nltk.download('punkt_tab')`."
                )
            self._tokenizer = tokenized_on_words
        else:
            self._tokenizer = partial(tokenize_on_ngrams, n=self.n)

    @property
    def inputs(self) -> List[str]:
        return ["text"]

    @property
    def outputs(self) -> List[str]:
        # Do we need to keep anything, or can it be stored in the cache?
        return ["hashvalues"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        tokenized_texts = []
        for input in inputs:
            tokenized_texts.append(self._tokenizer([input[self.inputs[0]]])[0])

        minhashes = self._hasher(
            tokenized_texts, num_perm=self.num_perm, seed=self.seed
        )
        for input, mh in zip(inputs, minhashes):
            input["hashvalues"] = mh.hashvalues
        yield inputs


class MinHashLSH(GlobalStep):
    """Creates a `MinHashLSH` index to deduplicate texts using MinHash.

    This class must be used together with `MinHash` step. It will work with the previous hashes
    to detect duplicate texts, and inform whether a given row can be removed.

    Attributes:
        seed: the seed to use for the MinHash. This seed must be the same
            used for `MinHash`, keep in mind when both steps are created. Defaults to `1`.
        num_perm: the number of permutations to use. Defaults to `128`.
        threshold: the threshold to consider two MinHashes as duplicates.
            Values closer to 0 detect more duplicates. Defaults to `0.9`.
        drop_hashvalues: whether to drop the hashvalues after processing. Defaults to `False`.
        storage: the storage to use for the LSH. Can be `dict` to store the index
            in memory, or `disk`, which uses a custom `shelve` backend. Note the `disk`
            is an experimetal feature that may cause issues. Defaults to `dict`.

    Input columns:
        - text (`str`): the texts to be filtered.
        - hashvalues (`List[int]`): hash values obtained from `MinHash` step.

    Output columns:
        - minhash_duplicate (`bool`): boolean indicating if the piece of text is a
            duplicate or not, so the user can decide afterwards whether to remove it
            or not.

    Categories:
        - filtering

    References:
        - [`datasketch documentation`](https://ekzhu.github.io/datasketch/lsh.html)

    Examples:

        Deduplicate a list of texts using MinHash and MinHashLSH:

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
    """

    seed: int = 1
    num_perm: int = 128
    threshold: float = 0.9
    drop_hashvalues: bool = False
    storage: Literal["dict", "disk"] = "dict"
    _lhs: Union["MinHashLSH", None] = PrivateAttr(None)
    _minhasher: Union["LeanMinHash", None] = PrivateAttr(None)

    def load(self) -> None:
        super().load()
        if not importlib.import_module("datasketch"):
            raise ImportError(
                "`datasketch` is needed to deduplicate with MinHash, but is not installed. "
                "Please install it using `pip install datasketch`."
            )
        from datasketch import LeanMinHash

        from distilabel.steps.filtering._datasketch import MinHashLSH

        self._lsh = MinHashLSH(
            num_perm=self.num_perm,
            threshold=self.threshold,
            storage_config={"type": self.storage},
        )
        self._minhasher = partial(LeanMinHash, seed=self.seed)

    def unload(self) -> None:
        super().unload()
        # In case of LSH being stored in disk, we need to close the file.
        if self.storage == "disk":
            self._lsh.close()

    @property
    def inputs(self) -> List[str]:
        return ["text", "hashvalues"]

    @property
    def outputs(self) -> List[str]:
        return ["keep_row_after_minhash_filtering"]

    def process(self, inputs: StepInput) -> "StepOutput":
        for input in inputs:
            minhash = self._minhasher(hashvalues=input["hashvalues"])
            # Check if the text is already in the LSH index
            if self._lsh.query(minhash):
                input["keep_row_after_minhash_filtering"] = False
            else:
                self._lsh.insert(str(uuid.uuid4()), minhash)
                input["keep_row_after_minhash_filtering"] = True
            if self.drop_hashvalues:
                del input["hashvalues"]

        yield inputs
