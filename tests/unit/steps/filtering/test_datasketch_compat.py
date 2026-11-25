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
from distilabel.steps.filtering.minhash import MinHashDedup


@pytest.mark.parametrize("storage", ["dict", "disk"])
@pytest.mark.parametrize("prepickle", [None, False, True])
def test_custom_lsh_accepts_python_keys(storage, prepickle, tmp_path, monkeypatch):
    monkeypatch.setattr(_datasketch, "KEY_VALUE_DISK_DIR", tmp_path)
    index = _datasketch.MinHashLSH(
        num_perm=32,
        threshold=0.9,
        storage_config={"type": storage},
        prepickle=prepickle,
    )
    signature = MinHash(num_perm=32)
    signature.update(b"training data")
    try:
        for key in ["document", 7, b"bytes"]:
            index.insert(key, signature)
        assert set(index.query(signature)) == {"document", 7, b"bytes"}
        assert all(key in index for key in ["document", 7, b"bytes"])
    finally:
        index.close()


@pytest.mark.parametrize("storage", ["dict", "disk"])
def test_minhash_dedup_across_batches(storage, tmp_path, monkeypatch):
    monkeypatch.setattr(_datasketch, "KEY_VALUE_DISK_DIR", tmp_path)
    step = MinHashDedup(
        tokenizer="ngrams", n=3, num_perm=32, threshold=0.9, storage=storage
    )
    step.load()
    try:
        first = next(step.process([{"text": "abcdef"}, {"text": "uvwxyz"}]))
        second = next(step.process([{"text": "abcdef"}, {"text": "uvwxyz"}]))
        assert [row["keep_row_after_minhash_filtering"] for row in first] == [
            True,
            True,
        ]
        assert [row["keep_row_after_minhash_filtering"] for row in second] == [
            False,
            False,
        ]
    finally:
        step.unload()
