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
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps import StepInput
from distilabel.steps.tasks import TextClassification
from distilabel.steps.tasks.base import GlobalTask
from distilabel.steps.typing import StepColumns
from distilabel.utils.itertools import batched

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class TextClustering(TextClassification, GlobalTask):
    """Task that clusters a set of texts and generates summary labels for each cluster.

    This is a `GlobalTask` that inherits from `TextClassification`, this means that all
    the attributes from that class are available here. Also, in this case we deal
    with all the inputs at once, instead of using batches. The `input_batch_size` is
    used here to send the examples to the LLM in batches (a subtle difference with the
    more common `Task` definitions).
    The task looks in each cluster for a given number of representative examples (the number
    is set by the `samples_per_cluster` attribute), and sends them to the LLM to get a label/s
    that represent the cluster. The labels are then assigned to each text in the cluster.
    The clusters and projections used in the step, are assumed to be obtained from the `UMAP`
    + `DBSCAN` steps, but could be generated for similar steps, as long as they represent the
    same concepts.
    This step runs a pipeline like the one in this repository:
    https://github.com/huggingface/text-clustering

    Input columns:
        - text (`str`): The reference text we want to obtain labels for.
        - projection (`List[float]`): Vector representation of the text to cluster,
            normally the output from the `UMAP` step.
        - cluster_label (`int`): Integer representing the label of a given cluster. -1
            means it wasn't clustered.

    Output columns:
        - summary_label (`str`): The label or list of labels for the text.
        - model_name (`str`): The name of the model used to generate the label/s.

    Categories:
        - clustering
        - text-classification

    References:
        - [`text-clustering repository`](https://github.com/huggingface/text-clustering)

    Attributes:
        - savefig: Whether to generate and save a figure with the clustering of the texts.
        - samples_per_cluster: The number of examples to use in the LLM as a sample of the cluster.

    Examples:
        Generate labels for a set of texts using clustering:

        ```python
        from distilabel.llms import InferenceEndpointsLLM
        from distilabel.steps import UMAP, DBSCAN, TextClustering
        from distilabel.pipeline import Pipeline

        ds_name = "argilla-warehouse/personahub-fineweb-edu-4-clustering-100k"

        with Pipeline(name="Text clustering dataset") as pipeline:
            batch_size = 500

            ds = load_dataset(ds_name, split="train").select(range(10000))
            loader = make_generator_step(ds, batch_size=batch_size, repo_id=ds_name)

            umap = UMAP(n_components=2, metric="cosine")
            dbscan = DBSCAN(eps=0.3, min_samples=30)

            text_clustering = TextClustering(
                llm=InferenceEndpointsLLM(
                    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
                ),
                n=3,  # 3 labels per example
                query_title="Examples of Personas",
                samples_per_cluster=10,
                context=(
                    "Describe the main themes, topics, or categories that could describe the "
                    "following types of personas. All the examples of personas must share "
                    "the same set of labels."
                ),
                default_label="None",
                savefig=True,
                input_batch_size=8,
                input_mappings={"text": "persona"},
                use_default_structured_output=True,
            )

            loader >> umap >> dbscan >> text_clustering
        ```
    """

    savefig: Optional[RuntimeParameter[bool]] = Field(
        default=True,
        description="Whether to generate and save a figure with the clustering of the texts.",
    )
    samples_per_cluster: int = Field(
        default=10,
        description="The number of examples to use in the LLM as a sample of the cluster.",
    )
    outputs: StepColumns = ["summary_label", "model_name"]

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.inputs = self.inputs + ["projection", "cluster_label"]

    def load(self) -> None:
        super().load()
        if self.savefig and (importlib.util.find_spec("matplotlib") is None):
            raise ImportError(
                "`matplotlib` package is not installed. Please install it using `pip install matplotlib`."
            )

    def _save_figure(
        self,
        data: pd.DataFrame,
        cluster_centers: Dict[str, Tuple[float, float]],
        cluster_summaries: Dict[int, str],
    ) -> None:
        """Saves the figure starting from the dataframe, using matplotlib.

        Args:
            data: pd.DataFrame with the columns 'X', 'Y' and 'labels' representing
                the projections and the label of each text respectively.
            cluster_centers: Dictionary mapping from each label the center of a cluster,
                to help with the placement of the annotations.
            cluster_summaries: The summaries of the clusters, obtained from the LLM.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        unique_labels = data["labels"].unique()
        # Map of colors for each label (-1 is black)
        colormap = dict(
            zip(unique_labels, plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))))
        )
        colormap[-1] = np.array([0, 0, 0, 0])
        data["color"] = data["labels"].map(colormap)

        data.plot(
            kind="scatter",
            x="X",
            y="Y",
            c="color",
            s=0.75,
            alpha=0.8,
            linewidth=0.4,
            ax=ax,
            colorbar=False,
        )

        for label in cluster_summaries.keys():
            if label == -1:
                continue
            summary = str(cluster_summaries[label])  # These are obtained from the LLM
            position = cluster_centers[label]
            t = ax.text(
                position[0],
                position[1],
                summary,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=4,
            )
            t.set_bbox(
                {
                    "facecolor": "white",
                    "alpha": 0.9,
                    "linewidth": 0,
                    "boxstyle": "square,pad=0.1",
                }
            )

        ax.set_axis_off()
        # Save the plot as an artifact of the step
        self.save_artifact(
            name="Text clusters",
            write_function=lambda path: fig.savefig(path / "figure_clustering.png"),
            metadata={"type": "image", "library": "matplotlib"},
        )
        plt.close()

    def _create_figure(
        self,
        inputs: StepInput,
        label2docs: Dict[int, List[str]],
        cluster_summaries: Dict[int, str],
    ) -> None:
        """Creates a figure of the clustered texts and save it as an artifact.

        Args:
            inputs: The inputs of the step, as we will extract information from them again.
            label2docs: Map from each label to the list of documents (texts) that belong to that cluster.
            cluster_summaries: The summaries of the clusters, obtained from the LLM.
        """
        self._logger.info("🖼️ Creating figure for the clusters...")

        labels = []
        projections = []
        id2cluster = {}
        for i, input in enumerate(inputs):
            label = input["cluster_label"]
            id2cluster[i] = label
            labels.append(label)
            projections.append(input["projection"])

        projections = np.array(projections)

        # Contains the placement of the cluster centers in the figure
        cluster_centers: Dict[str, Tuple[float, float]] = {}
        for label in label2docs.keys():
            x = np.mean([projections[doc, 0] for doc in label2docs[label]])
            y = np.mean([projections[doc, 1] for doc in label2docs[label]])
            cluster_centers[label] = (x, y)

        df = pd.DataFrame(
            data={
                "X": projections[:, 0],
                "Y": projections[:, 1],
                "labels": labels,
            }
        )

        self._save_figure(
            df, cluster_centers=cluster_centers, cluster_summaries=cluster_summaries
        )

    def _prepare_input_texts(
        self,
        inputs: StepInput,
        label2docs: Dict[int, List[int]],
        unique_labels: List[int],
    ) -> List[Dict[str, Union[str, int]]]:
        """Prepares a batch of inputs to send to the LLM, with the examples of each cluster.

        Args:
            inputs: Inputs from the step.
            label2docs: Map from each label to the list of documents (texts) that
                belong to that cluster.
            unique_labels: The unique labels of the clusters.

        Returns:
            The input texts to send to the LLM, with the examples of each cluster
            prepared to be used in the prompt, and an additional key to store the
            labels (that will be needed to find the data after the batches are
            returned from the LLM).
        """
        input_texts = []
        for label in range(unique_labels):  # The label -1 is implicitly excluded
            # Get the ids but remove possible duplicates, which could happen with bigger probability
            # the bigger the number of examples requested, and the smaller the subset of examples
            ids = set(
                np.random.choice(label2docs[label], size=self.samples_per_cluster)
            )  # Grab the number of examples
            examples = [inputs[i]["text"] for i in ids]
            input_text = {
                "text": "\n\n".join(
                    [f"Example {i}:\n{t}" for i, t in enumerate(examples, start=1)]
                ),
                "__LABEL": label,
            }
            input_texts.append(input_text)
        return input_texts

    def process(self, inputs: StepInput) -> "StepOutput":
        labels = [input["cluster_label"] for input in inputs]
        # -1 because -1 is the label for the unclassified
        unique_labels = len(set(labels)) - 1
        # This will be the output of the LLM, the set of labels for each cluster
        cluster_summaries: Dict[int, str] = {-1: self.default_label}

        # Map from label to list of documents, will use them to select examples from each cluster
        label2docs = defaultdict(list)
        for i, label in enumerate(labels):
            label2docs[label].append(i)

        input_texts = self._prepare_input_texts(inputs, label2docs, unique_labels)

        # Send the texts in batches to the LLM, and get the labels for each cluster
        for i, batched_inputs in enumerate(batched(input_texts, self.input_batch_size)):
            self._logger.info(f"📦 Processing internal batch of inputs {i}...")
            results = super().process(batched_inputs)
            for result in next(results):  # Extract the elements from the generator
                cluster_summaries[result["__LABEL"]] = result["labels"]

        # Assign the labels to each text
        for input in inputs:
            input["summary_label"] = json.dumps(
                cluster_summaries[input["cluster_label"]]
            )

        if self.savefig:
            self._create_figure(inputs, label2docs, cluster_summaries)

        yield inputs
