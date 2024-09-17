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

from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, MinHashDedup


def test_minhash_deduplication() -> None:
    with Pipeline() as pipeline:
        ds_size = 1000
        batch_size = 500
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
        minhash = MinHashDedup(
            tokenizer="ngrams",
            n=2,
            threshold=0.9,
            storage="disk",
            input_batch_size=batch_size,
        )
        data >> minhash

    distiset = pipeline.run(use_cache=False)
    ds = distiset["default"]["train"]
    ds_dedup = ds.filter(lambda x: x["keep_row_after_minhash_filtering"])
    assert len(ds_dedup) == 4


if __name__ == "__main__":
    test_minhash_deduplication()
