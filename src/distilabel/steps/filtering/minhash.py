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
from itertools import tee
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Union,
)

from nltk.tokenize import word_tokenize
from pydantic import PrivateAttr
from typing_extensions import override

from distilabel.steps.base import GlobalStep, Step, StepInput

if TYPE_CHECKING:
    from datasketch import LeanMinHash, MinHash, MinHashLSH

    from distilabel.steps.typing import StepOutput


# Copied from: https://github.com/huggingface/datatrove/blob/main/src/datatrove/utils/text.py#L89C1-L95C65
def ngrams(sequence: Iterable, n: int):
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
        Iterable[Set[bytes]]: List with the set of tokens for each document.
    """
    return [{w.encode("utf-8") for w in word_tokenize(text)} for text in texts]


def tokenize_on_ngrams(texts: Iterable[str], n: int = 1) -> List[Set[bytes]]:
    """Tokenizes a list of texts into ngrams, and returns the set of them as bytes.

    Args:
        texts: List of documents to be tokenized.
        n: The size of the ngrams, defaults to 1 (single letters).

    Returns:
        Iterable[Set[bytes]]: List with the set of tokens for each document.
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
    """
    Attributes:
        column: the column to deduplicate. Defaults to `text`.
        num_perm: the number of permutations to use. Defaults to `128`.
        seed: the seed to use for the MinHash. Defaults to `1`.
        tokenizer: the tokenizer to use. Defaults to `ngrams`.
        n: the size of the ngrams to use. Only relevant if `tokenizer="ngrams"`, Defaults to `1`.

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
        minhasher = MinHash(tokenizer="words")
        minhasher.load()
        result = next(hasher.process([{"text": t} for t in texts]))
        ```
    """

    column: str = "text"
    num_perm: int = 128
    seed: int = 1
    tokenizer: Literal["words", "ngrams"] = "ngrams"
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
        from functools import partial

        self._tokenizer = (
            tokenized_on_words
            if self.tokenizer == "words"
            else partial(tokenize_on_ngrams, n=self.n)
        )

    @property
    def inputs(self) -> List[str]:
        return [self.column]

    @property
    def outputs(self) -> List[str]:
        # Do we need to keep anything, or can it be stored in the cache?
        return ["hashvalues"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        tokenized_texts = []
        for input in inputs:
            tokenized_texts.append(self._tokenizer([input[self.column]])[0])

        minhashes = self._hasher(
            tokenized_texts, num_perm=self.num_perm, seed=self.seed
        )
        for input, mh in zip(inputs, minhashes):
            input["hashvalues"] = mh.hashvalues
        yield inputs


class MinHashLSH(GlobalStep):
    """
    This class must be used together with MinHash. It creates hashes internally.

    Attributes:
        seed: the seed to use for the MinHash. Defaults to `1`.
        num_perm: the number of permutations to use. Defaults to `128`.
        threshold: the threshold to consider two MinHashes as duplicates. Defaults to `0.9`.

    Examples:

        Deduplicate a list of texts using MinHash and MinHashLSH:

        ```python
        from distilabel.steps import MinHash, MinHashLSH

        texts: List[str] = [
            "This is a test document.",
            "This document is a test.",
            "Test document for duplication.",
            "Document for duplication test.",
            "This is another unique document."
        ]

        minhasher = MinHash(tokenizer="words")
        minhasher.load()
        results_with_hashes = next(minhasher.process([{"text": t} for t in texts]))

        minhash_lsh = MinHashLSH(threshold=09, seed=hasher.seed)
        minhash_lsh.load()
        result = next(minhash_lsh.process(results_with_hashes))
        ```
    """

    seed: int = 1
    num_perm: int = 128
    threshold: float = 0.9
    drop_hashvalues: bool = False
    _lhs: Union["MinHashLSH", None] = PrivateAttr(None)
    _minhasher: Union["LeanMinHash", None] = PrivateAttr(None)

    def load(self) -> None:
        super().load()
        if not importlib.import_module("datasketch"):
            raise ImportError(
                "`datasketch` is needed to deduplicate with MinHash, but is not installed. "
                "Please install it using `pip install datasketch`."
            )
        from datasketch import LeanMinHash, MinHashLSH

        self._lsh = MinHashLSH(num_perm=self.num_perm, threshold=self.threshold)
        from functools import partial

        self._minhasher = partial(LeanMinHash, seed=self.seed)

    @property
    def inputs(self) -> List[str]:
        return ["text", "hashvalues"]

    @property
    def outputs(self) -> List[str]:
        return ["minhash_duplicate"]

    def process(self, inputs: StepInput) -> "StepOutput":
        for input in inputs:
            minhash = self._minhasher(hashvalues=input["hashvalues"])
            # Check if the text is already in the LSH index
            if self._lsh.query(minhash):
                input["minhash_duplicate"] = True
            else:
                self._lsh.insert(str(uuid.uuid4()), minhash)
                input["minhash_duplicate"] = False
            if self.drop_hashvalues:
                del input["hashvalues"]

        yield inputs
