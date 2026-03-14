---
hide: toc
---
# Text clustering pipeline with `distilabel`

Build a text clustering pipeline with [`EmbeddingGeneration`](../../../components-gallery/steps/embeddinggeneration.md), [`UMAP`](../../../components-gallery/steps/umap.md), [`DBSCAN`](../../../components-gallery/steps/dbscan.md), and [`TextClustering`](../../../components-gallery/tasks/textclustering.md).

The example also includes an optional branch using [`FaissNearestNeighbour`](../../../components-gallery/steps/faissnearestneighbour.md) to retrieve semantic nearest neighbours for each sample.

To run this example, install the optional dependencies for embeddings, clustering, and nearest neighbours:

```console
pip install "distilabel[sentence-transformers,text-clustering,hf-inference-endpoints,faiss-cpu]"
```

??? Run

    ```python
    python examples/text_clustering.py
    ```

```python title="text_clustering.py"
--8<-- "examples/text_clustering.py"
```
