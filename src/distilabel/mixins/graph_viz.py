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

try:
    import graphviz
except ImportError as e:
    raise ImportError(
        "graphviz is not installed. Please install it using"
        " `pip install graphviz`, and follow the instructions at:"
        " https://graphviz.readthedocs.io/en/stable/manual.html#installation"
    ) from e

import textwrap
from pathlib import Path
from typing import Dict, Optional


class GraphvizMixin:
    """Mixin for `BasePipeline` to visualize the underlying `DAG` graph using `graphviz`.

    Attributes:
        _dot: The `graphviz.Digraph` object to store the graph visualization.
    """

    _dot: Optional[graphviz.Digraph] = None
    # The filename will be used in case we want to find the easily afterwards
    _graph_filename: Optional[str] = None

    def create_graphviz(self) -> None:
        """Creates a graph visualization of the pipeline."""
        self._dot = graphviz.Digraph(comment=self.name, filename=self.name, strict=True)

        def label_info(step):
            # Extracts the relevant info from the step to be used as a label
            not_required = {
                "name",
                "input_mappings",
                "output_mappings",
                "type_info",
                "runtime_parameters_info",
            }
            dump = step.dump()
            # More info in the '\l': https://graphviz.org/docs/attrs/nojustify/
            name = dump.pop("name") + r"\l"
            for key in not_required:
                dump.pop(key, None)

            params = ""
            for key, value in dump.items():
                # If the step is an LLM, get just the model name to avoid long strings,
                # otherwise, shorten the string to 30 characters max.
                value = (
                    step.llm.model_name
                    if key == "llm"
                    else textwrap.shorten(str(value), width=30)
                )
                params += rf"    - {key}: {value}\l"

            return name + "".join(params)

        # Import here to avoid circular imports
        from distilabel.steps.base import GeneratorStep, GlobalStep
        from distilabel.steps.task.base import Task

        def get_fillcolor(step):
            # Use different colors depending on the type of Step
            if isinstance(step, Task):
                return "#fff0f6"
            if isinstance(step, GlobalStep):
                return "#e3fafc"
            if isinstance(step, GeneratorStep):
                return "#f3f0ff"
            return "#fff5f5"

        prepare_label = True
        for node_name in self.dag:
            step = self.dag.get_step(node_name)["step"]
            label = label_info(step) if prepare_label else node_name
            self._dot.node(
                node_name,
                label=label,
                style="rounded,filled",
                fillcolor=get_fillcolor(step),
                nojustify="true",
            )

        for edge_tail, edge_head in self.dag.G.edges:
            self._dot.edge(edge_tail, edge_head, constraint="true")

    def apply_visualization_style(
        self,
        graph_attrs: Optional[Dict[str, str]] = None,
        node_attrs: Optional[Dict[str, str]] = None,
        edge_attrs: Optional[Dict[str, str]] = None,
    ) -> None:
        """Applies the visualization style to the graph.

        Args:
            graph_attrs: The attributes to apply to the graph.
            node_attrs: The attributes to apply to the nodes.
            edge_attrs: The attributes to apply to the edges.
        """

        if graph_attrs is not None:
            self._dot.graph_attr.update(**graph_attrs)
        else:
            self._dot.graph_attr.update(
                {
                    # Sets the direction from Top to Bottom
                    "rankdir": "TB",
                    "splines": "ortho",
                    # The name of the Pipeline that will be shown on the top of the graph
                    "label": self.name,
                    "pad": "0.5",
                    "nodesep": "0.60",
                    "ranksep": "0.5",
                    "fontname": "Sans-Serif",
                    "fontsize": "20",
                    "fontcolor": "#2D3436",
                    "labelloc": "t",
                }
            )
        if node_attrs is not None:
            self._dot.node_attr.update(**node_attrs)
        else:
            self._dot.node_attr.update(
                {
                    "shape": "box",
                    "fixedsize": "false",
                    "width": "0.4",
                    "height": "0.25",
                    "labelloc": "c",
                    "imagescale": "true",
                    "fontname": "Sans-Serif",
                    "fontsize": "13",
                    "fontcolor": "#000000",
                }
            )
        if edge_attrs is not None:
            self._dot.edge_attr.update(**edge_attrs)
        else:
            self._dot.edge_attr.update(
                {
                    "color": "#999999",
                }
            )

    def render(
        self, view: bool = False, format_: str = "png", cleanup: bool = True
    ) -> None:
        """Creates the graph and saves it to a file.

        Args:
            view: Whether to automatically open the file. Defaults to False.
            format_: File format of the rendered graph. Defaults to "png".
            cleanup: Whether to remove the files created in the folder other than the figure.
                Defaults to True.
        """
        # NOTE: Should we allow the user to specify the path?
        if self._dot is None:
            self.create_graphviz()
            self.apply_visualization_style()
        formatted_name = self.name.replace(" ", "_")
        # Write the file to the cache location
        self._graph_filename = (
            self._cache_location["pipeline"].parent
            / f"distilabel-dag-graph/{formatted_name}"
        )
        self._dot.render(self._graph_filename, view=view, format=format_)
        if cleanup:
            for file in Path(self._graph_filename).parent.iterdir():
                if file.name != f"{formatted_name}.{format_}":
                    file.unlink()
