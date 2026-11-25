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

import pytest
from datasketch import MinHash

from distilabel.steps.filtering import _datasketch
from distilabel.steps.filtering._datasketch import MinHashLSH
from distilabel.steps.filtering.minhash import MinHashDedup, tokenize_on_ngrams


@pytest.mark.parametrize("storage", ["dict", "disk"])
@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize("num_perm", [32, 128])
@pytest.mark.parametrize("threshold", [0.2, 0.9])
def test_streamed_minhashes_match_bulk(
    seed, num_perm, threshold, storage, tmp_path, monkeypatch
):
    monkeypatch.setattr(_datasketch, "KEY_VALUE_DISK_DIR", tmp_path)
    texts = [
        "",
        "a",
        "a",
        "training corpus example",
        "training corpus examples",
        "completely different text",
        "数学训练数据",
        "数学训练数据",
        "training corpus example",
    ]
    signatures = MinHash.bulk(
        tokenize_on_ngrams(texts, n=3), seed=seed, num_perm=num_perm
    )
    reference = MinHashLSH(
        threshold=threshold, num_perm=num_perm, storage_config={"type": storage}
    )
    expected = []
    for i, signature in enumerate(signatures):
        keep = not reference.query(signature)
        expected.append(keep)
        if keep:
            reference.insert(str(i), signature)

    step = MinHashDedup(
        tokenizer="ngrams",
        n=3,
        seed=seed,
        num_perm=num_perm,
        threshold=threshold,
        storage=storage,
    )
    step.load()
    actual_signatures = []
    hasher = step._hasher

    def capture(tokens, **kwargs):
        for signature in hasher(tokens, **kwargs):
            actual_signatures.append(signature.hashvalues.copy())
            yield signature

    step._hasher = capture
    output = []
    try:
        # Duplicates cross both within-batch and between-batch boundaries.
        for chunk in (texts[:4], texts[4:7], texts[7:], []):
            output.extend(next(step.process([{"text": text} for text in chunk])))
        assert [row["text"] for row in output] == texts
        assert [row["keep_row_after_minhash_filtering"] for row in output] == expected
        assert all(
            (actual == expected.hashvalues).all()
            for actual, expected in zip(actual_signatures, signatures)
        )
        assert len(actual_signatures) == len(signatures)
    finally:
        step.unload()
        reference.close()


def test_index_updated_before_next_document_is_tokenized():
    step = MinHashDedup(tokenizer="ngrams", n=3)
    step.load()
    original = step._tokenizer
    tokenized = 0

    def tokenize(texts):
        nonlocal tokenized
        # Distinct texts are inserted before the next document is tokenized.
        assert len(step._lsh.keys) == tokenized
        tokenized += 1
        return original(texts)

    step._tokenizer = tokenize
    try:
        output = next(step.process([{"text": "abcdef"}, {"text": "uvwxyz"}]))
        assert all(row["keep_row_after_minhash_filtering"] for row in output)
        assert tokenized == 2
    finally:
        step.unload()
