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

import importlib.util
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from pydantic import Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps import (
    GlobalStep,
    StepInput,
)

if TYPE_CHECKING:
    from sklearn.cluster import DBSCAN as _DBSCAN

    from distilabel.steps.typing import StepOutput


class DBSCAN(GlobalStep):
    r"""DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core
    samples in regions of high density and expands clusters from them. This algorithm
    is good for data which contains clusters of similar density.

    This is a `GlobalStep` that clusters the embeddings using the DBSCAN algorithm
    from `sklearn`. Visit `TextClustering` step for an example of use.
    The trained model is saved as an artifact when creating a distiset
    and pushing it to the Hugging Face Hub.

    Input columns:
        - projection (`List[float]`): Vector representation of the text to cluster,
            normally the output from the `UMAP` step.

    Output columns:
        - cluster_label (`int`): Integer representing the label of a given cluster. -1
            means it wasn't clustered.

    Categories:
        - clustering
        - text-classification

    References:
        - [`DBSCAN demo of sklearn`](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#demo-of-dbscan-clustering-algorithm)
        - [`sklearn dbscan`](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

    Attributes:
        - eps: The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is not a maximum bound on the distances of
            points within a cluster. This is the most important DBSCAN parameter to
            choose appropriately for your data set and distance function.
        - min_samples: The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself. If `min_samples`
            is set to a higher value, DBSCAN will find denser clusters, whereas if it is set
            to a lower value, the found clusters will be more sparse.
        - metric: The metric to use when calculating distance between instances in a feature
            array. If metric is a string or callable, it must be one of the options allowed
            by `sklearn.metrics.pairwise_distances` for its metric parameter.
        - n_jobs: The number of parallel jobs to run.

    Runtime parameters:
        - `eps`: The maximum distance between two samples for one to be considered as in the
            neighborhood of the other. This is not a maximum bound on the distances of
            points within a cluster. This is the most important DBSCAN parameter to
            choose appropriately for your data set and distance function.
        - `min_samples`: The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself. If `min_samples`
            is set to a higher value, DBSCAN will find denser clusters, whereas if it is set
            to a lower value, the found clusters will be more sparse.
        - `metric`: The metric to use when calculating distance between instances in a feature
            array. If metric is a string or callable, it must be one of the options allowed
            by `sklearn.metrics.pairwise_distances` for its metric parameter.
        - `n_jobs`: The number of parallel jobs to run.
    """

    eps: Optional[RuntimeParameter[float]] = Field(
        default=0.3,
        description=(
            "The maximum distance between two samples for one to be considered "
            "as in the neighborhood of the other. This is not a maximum bound "
            "on the distances of points within a cluster. This is the most "
            "important DBSCAN parameter to choose appropriately for your data set "
            "and distance function."
        ),
    )
    min_samples: Optional[RuntimeParameter[int]] = Field(
        default=30,
        description=(
            "The number of samples (or total weight) in a neighborhood for a point to "
            "be considered as a core point. This includes the point itself. If "
            "`min_samples` is set to a higher value, DBSCAN will find denser clusters, "
            "whereas if it is set to a lower value, the found clusters will be more "
            "sparse."
        ),
    )
    metric: Optional[RuntimeParameter[str]] = Field(
        default="euclidean",
        description=(
            "The metric to use when calculating distance between instances in a "
            "feature array. If metric is a string or callable, it must be one of "
            "the options allowed by `sklearn.metrics.pairwise_distances` for "
            "its metric parameter."
        ),
    )
    n_jobs: Optional[RuntimeParameter[int]] = Field(
        default=8, description="The number of parallel jobs to run."
    )

    _clusterer: Optional["_DBSCAN"] = PrivateAttr(None)

    def load(self) -> None:
        super().load()
        if importlib.util.find_spec("sklearn") is None:
            raise ImportError(
                "`sklearn` package is not installed. Please install it using `pip install scikit-learn`."
            )
        from sklearn.cluster import DBSCAN as _DBSCAN

        self._clusterer = _DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def unload(self) -> None:
        self._clusterer = None

    @property
    def inputs(self) -> List[str]:
        return ["projection"]

    @property
    def outputs(self) -> List[str]:
        return ["cluster_label"]

    def _save_model(self, model: Any) -> None:
        import joblib

        def save_model(path):
            with open(str(path / "DBSCAN.joblib"), "wb") as f:
                joblib.dump(model, f)

        self.save_artifact(
            name="DBSCAN_model",
            write_function=lambda path: save_model(path),
            metadata={
                "eps": self.eps,
                "min_samples": self.min_samples,
                "metric": self.metric,
            },
        )

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        projections = np.array([input["projection"] for input in inputs])

        self._logger.info("🏋️‍♀️ Start training DBSCAN...")
        fitted_clusterer = self._clusterer.fit(projections)
        cluster_labels = fitted_clusterer.labels_
        # Sets the cluster labels for each input, -1 means it wasn't clustered
        for input, cluster_label in zip(inputs, cluster_labels):
            input["cluster_label"] = cluster_label
        self._logger.info(f"DBSCAN labels assigned: {len(set(cluster_labels))}")
        self._save_model(fitted_clusterer)
        yield inputs
