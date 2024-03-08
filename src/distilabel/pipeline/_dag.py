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

import inspect
from collections import defaultdict
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Set,
    Type,
    Union,
)

import networkx as nx

from distilabel.utils.serialization import TYPE_INFO_KEY, _get_class, _Serializable

if TYPE_CHECKING:
    from distilabel.steps.base import _Step


class DAG(_Serializable):
    """A Directed Acyclic Graph (DAG) to represent the pipeline.

    Attributes:
        G: The graph representing the pipeline.
    """

    def __init__(self) -> None:
        self.G = nx.DiGraph()

    def __iter__(self) -> Generator[str, None, None]:
        yield from self.G

    def __len__(self) -> int:
        return len(self.G)

    def add_step(self, step: "_Step") -> None:
        """Add a step to the DAG.

        Args:
            step: The step to add to the DAG.

        Raises:
            ValueError: If a step with the same name already exists in the DAG.
        """
        name = step.name
        if name in self.G:
            raise ValueError(f"Step with name '{name}' already exists")
        self.G.add_node(name, step=step)

    def get_step(self, name: str) -> Dict[str, Any]:
        """Get a step from the DAG.

        Args:
            name: The name of the step to get.

        Returns:
            The step with the given name.

        Raises:
            ValueError: If the step with the given name does not exist.
        """
        if name not in self.G:
            raise ValueError(f"Step with name '{name}' does not exist")
        return self.G.nodes[name]

    def set_step_attr(self, name: str, attr: str, value: Any) -> None:
        """Set an attribute of a step in the DAG.

        Args:
            name: The name of the step.
            attr: The attribute to set.
            value: The value to set.

        Raises:
            ValueError: If the step with the given name does not exist.
        """
        if name not in self.G:
            raise ValueError(f"Step with name '{name}' does not exist")
        self.G.nodes[name][attr] = value

    def add_edge(self, from_step: str, to_step: str) -> None:
        """Add an edge between two steps in the DAG.

        Args:
            from_step: The name of the step from which the edge starts.
            to_step: The name of the step to which the edge ends.

        Raises:
            ValueError: If the edge cannot be added.
        """
        if from_step not in self.G:
            raise ValueError(f"Step with name '{from_step}' does not exist")

        if to_step not in self.G:
            raise ValueError(f"Step with name '{to_step}' does not exist")

        if to_step in self.G[from_step]:
            raise ValueError(
                f"There is already a edge from '{to_step}' to '{from_step}'"
            )

        if to_step in nx.ancestors(self.G, from_step):
            raise ValueError(
                f"Cannot add edge from '{from_step}' to '{to_step}' as it would create a cycle."
            )

        self.G.add_edge(from_step, to_step)

    @cached_property
    def root_steps(self) -> Set[str]:
        """The steps that don't have any predecessors i.e. generator steps.

        Returns:
            A list with the names of the steps that don't have any predecessors.
        """
        return {node for node, degree in self.G.in_degree() if degree == 0}

    @cached_property
    def leaf_steps(self) -> Set[str]:
        """The steps that don't have any successors.

        Returns:
            A list with the names of the steps that don't have any successors.
        """
        return {node for node, degree in self.G.out_degree() if degree == 0}

    def get_step_predecessors(self, step_name: str) -> Iterable[str]:
        """Gets the predecessors of a step.

        Args:
            step_name: The name of the step.

        Returns:
            An iterable with the names of the steps that are predecessors of the given step.
        """
        if step_name not in self.G:
            raise ValueError(f"Step '{step_name}' does not exist")
        return self.G.predecessors(step_name)

    def get_step_successors(self, step_name: str) -> Iterable[str]:
        """Gets the successors of a step.

        Args:
            step_name: The name of the step.

        Returns:
            An iterable with the names of the steps that are successors of the given step.
        """

        if step_name not in self.G:
            raise ValueError(f"Step '{step_name}' does not exist")
        return self.G.successors(step_name)

    def iter_based_on_trophic_levels(self) -> Iterable[List[str]]:
        """Iterate over steps names in the DAG based on their trophic levels. This is similar
        to a topological sort, but we also know which steps are at the same level and
        can be run in parallel.

        Yields:
            A list containing the names of the steps that can be run in parallel.
        """
        trophic_levels = nx.trophic_levels(self.G)

        v = defaultdict(list)
        for step, trophic_level in trophic_levels.items():
            v[int(trophic_level)].append(step)

        for trophic_level in sorted(v.keys()):
            yield v[trophic_level]

    def validate(self) -> None:
        """Validates that the `Step`s included in the pipeline are correctly connected and
        have the correct inputs and outputs.

        Raises:
            ValueError: If the pipeline is not valid.
        """

        for trophic_level, steps in enumerate(
            self.iter_based_on_trophic_levels(), start=1
        ):
            for step_name in steps:
                step: "_Step" = self.get_step(step_name)["step"]

                step.verify_inputs_mappings()
                step.verify_outputs_mappings()
                self._validate_step_process_arguments(step)

                # Validate that the steps in the first trophic level are `GeneratorStep`s
                if trophic_level == 1:
                    if not step.is_generator:
                        raise ValueError(
                            f"Step '{step_name}' should be `GeneratorStep` as it doesn't"
                            " have any previous steps"
                        )
                else:
                    self._step_inputs_are_available(step)

    def _step_inputs_are_available(self, step: "_Step") -> None:
        """Validates that the `Step.inputs` will be available when the step gets to be
        executed in the pipeline i.e. the step will receive list of dictionaries containing
        its inputs as keys.

        Args:
            step: The step.
        """
        inputs_available_for_step = [
            output
            for step_name in nx.ancestors(self.G, step.name)
            for output in self.get_step(step_name)["step"].get_outputs()
        ]
        step_inputs = step.get_inputs()
        if not all(input in inputs_available_for_step for input in step_inputs):
            raise ValueError(
                f"Step '{step.name}' requires inputs {step_inputs} which are not"
                f" available when the step gets to be executed in the pipeline."
                f" Please make sure previous steps to '{step.name}' are generating"
                f" the required inputs. Available inputs are: {inputs_available_for_step}"
            )

    def _validate_step_process_arguments(self, step: "_Step") -> None:
        """Validates the arguments of the `Step.process` method, checking there is an
        argument with type hint `StepInput` and that all the required runtime parameters
        are provided.

        Args:
            step: The step to validate.

        Raises:
            ValueError: If the arguments of the `process` method of the step are not valid.
        """

        step_input_parameter = step.get_process_step_input()
        self._validate_process_step_input_parameter(step.name, step_input_parameter)
        self._validate_step_process_runtime_parameters(step)

    def _validate_process_step_input_parameter(
        self,
        step_name: str,
        step_input_parameter: Union[inspect.Parameter, None] = None,
    ) -> None:
        """Validates that the `Step.process` method has a parameter with type hint `StepInput`

        Args:
            step_name: The name of the step.
            step_input_parameter: The parameter with type hint `StepInput` of the `process`
                method of the step.

        Raises:
            ValueError: If the `step_input_parameter` is not valid.
        """

        predecessors = {
            step_name: self.get_step(step_name)["step"]
            for step_name in self.G.predecessors(step_name)
        }
        num_predecessors = len(predecessors)

        if num_predecessors == 0:
            return

        if step_input_parameter is None:
            if num_predecessors > 1:
                prev_steps = ", ".join([f"'{step_name}'" for step_name in predecessors])
                raise ValueError(
                    f"Step '{step_name}' should have a `*args` parameter with type hint"
                    f" `StepInput` to receive outputs from previous steps: {prev_steps}."
                )

            prev_step_name = next(iter(predecessors))
            raise ValueError(
                f"Step '{step_name}' should have a parameter with type hint `StepInput`"
                f" to receive the output from the previous step: '{prev_step_name}'."
            )

        if (
            num_predecessors > 1
            and step_input_parameter.kind != inspect.Parameter.VAR_POSITIONAL
        ):
            raise ValueError(
                f"Step '{step_name}' should have a `*args` parameter with type hint `StepInput`"
                f" to receive outputs from previous steps."
            )

    def _validate_step_process_runtime_parameters(self, step: "_Step") -> None:
        """Validates that the required runtime parameters of the step are provided.

        Args:
            step: The step to validate.

        Raises:
            ValueError: If not all the required runtime parameters haven't been provided
                with a value.
        """
        runtime_parameters_values = step._runtime_parameters
        for param_name, has_default_value in step.runtime_parameters_names.items():
            if param_name not in runtime_parameters_values and not has_default_value:
                raise ValueError(
                    f"Step '{step.name}' is missing required runtime parameter '{param_name}'."
                    " Please, provide a value for it when calling `Pipeline.run`"
                )

    def _model_dump(self, obj: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the content of the DAG to a dict.

        References:
        * [`adjacency_data` - NetworkX Documentation](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.adjacency_data.html#networkx.readwrite.json_graph.adjacency_data)

        Args:
            obj (Any): Unused, just kept to match the signature of the parent method.
            kwargs (Any): Additional arguments that could be passed to the networkx function.

        Returns:
            Dict[str, Any]: Internal representation of the DAG from networkx in a serializable format.
        """
        from networkx.readwrite import json_graph

        adjacency_data = json_graph.adjacency_data(self.G, **kwargs)

        data = {"steps": [], "connections": []}
        for i, node in enumerate(adjacency_data["nodes"]):
            name = node["id"]
            data["steps"].append({"step": node["step"].dump(), "name": name})
            data["connections"].append(
                {
                    "from": name,
                    "to": [node["id"] for node in adjacency_data["adjacency"][i]],
                }
            )

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAG":
        """Generates the DAG from a dictionary with the steps serialized.

        Args:
            data (Dict[str, Any]): Dictionary with the serialized content (the content from self.dump()).

        Returns:
            DAG: Instance of the DAG from the serialized content.
        """

        dag = cls()

        for step in data["steps"]:
            cls_step: Type["_Step"] = _get_class(**step["step"][TYPE_INFO_KEY])
            dag.add_step(cls_step.from_dict(step["step"]))

        for connection in data["connections"]:
            from_step = connection["from"]
            for to_step in connection["to"]:
                dag.add_edge(from_step, to_step)

        return dag
