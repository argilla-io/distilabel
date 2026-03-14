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

from datasets import load_dataset

from distilabel.models import InferenceEndpointsLLM, SentenceTransformerEmbeddings
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    DBSCAN,
    UMAP,
    EmbeddingGeneration,
    FaissNearestNeighbour,
    TextClustering,
    make_generator_step,
)

ds_name = "ag_news"
dataset = load_dataset(ds_name, split="train").select(range(2000))

with Pipeline(name="text-clustering-pipeline") as pipeline:
    loader = make_generator_step(dataset=dataset, batch_size=200, repo_id=ds_name)

    embeddings = EmbeddingGeneration(
        embeddings=SentenceTransformerEmbeddings(
            model="mixedbread-ai/mxbai-embed-large-v1",
        )
    )

    umap = UMAP(n_components=2, metric="cosine")
    dbscan = DBSCAN(eps=0.15, min_samples=10)

    text_clustering = TextClustering(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        ),
        n=3,
        query_title="Examples of news",
        samples_per_cluster=10,
        context=(
            "Describe the main themes, topics, or categories shared by these news "
            "examples. All samples in the same cluster must share the same set of labels."
        ),
        default_label="Unclassified",
        input_batch_size=8,
        use_default_structured_output=True,
    )

    # Optional branch to inspect semantic neighbours for each sample.
    nearest_neighbours = FaissNearestNeighbour(k=3)

    loader >> embeddings
    embeddings >> umap >> dbscan >> text_clustering
    embeddings >> nearest_neighbours


if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    distiset.push_to_hub("USERNAME/text-clustering-example")
