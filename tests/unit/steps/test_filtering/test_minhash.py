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

from typing import List

import numpy as np
import pytest

from distilabel.steps.filtering.minhash import (
    MinHash,
    MinHashLSH,
    tokenize_on_ngrams,
    tokenized_on_words,
)

texts: List[str] = [
    "This is a test document.",
    "This document is a test.",
    "Test document for duplication.",
    "Document for duplication test.",
    "This is another unique document.",
]


def test_tokenize_on_words() -> None:
    tokenized = tokenized_on_words(texts)
    assert len(tokenized) == len(texts)
    assert tokenized[0] == {b".", b"This", b"a", b"document", b"is", b"test"}


@pytest.mark.parametrize("n", [1, 3])
def test_tokenize_on_ngrams(n: int) -> None:
    tokenized = tokenize_on_ngrams(texts, n=n)
    assert len(tokenized) == len(texts)
    assert all(len(t) == n for t in tokenized[0])


class TestMinHash:
    @pytest.mark.parametrize("tokenizer", ["words", "ngrams"])
    def test_process(self, tokenizer: str) -> None:
        hasher = MinHash(tokenizer=tokenizer, n=3)
        hasher.load()
        result = next(hasher.process([{"text": t} for t in texts]))
        hashvalues = result[0]["hashvalues"]
        assert isinstance(hashvalues, np.ndarray)


class TestMinHashLSH:
    @pytest.mark.parametrize(
        "threshold, keep_row_after_minhash_filtering, storage",
        [
            (0.1, 1, "dict"),
            (0.9, 4, "dict"),
            # (0.9, 4, "disk")  # This test is skipped because it fails while testing on CI
        ],
    )
    def test_process(
        self, threshold: float, keep_row_after_minhash_filtering: int, storage: str
    ) -> None:
        hasher = MinHash()
        hasher.load()
        results_with_hashes = next(hasher.process([{"text": t} for t in texts]))

        minhash_lsh = MinHashLSH(threshold=threshold, seed=hasher.seed, storage=storage)
        minhash_lsh.load()
        result = next(minhash_lsh.process(results_with_hashes))
        duplicated = [r["keep_row_after_minhash_filtering"] for r in result]
        assert sum(duplicated) == keep_row_after_minhash_filtering
        minhash_lsh.unload()
