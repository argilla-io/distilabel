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

from distilabel.pipeline.constants import (
    CONVERGENCE_STEP_ATTR_NAME,
    ROUTING_BATCH_FUNCTION_ATTR_NAME,
    STEP_ATTR_NAME,
)
from distilabel.pipeline.routing_batch_function import RoutingBatchFunction
from distilabel.steps.base import GeneratorStep
from distilabel.utils.serialization import (
    TYPE_INFO_KEY,
    _get_module_attr,
    _Serializable,
)

if TYPE_CHECKING:
    from distilabel.mixins.runtime_parameters import RuntimeParametersNames
    from distilabel.steps.base import Step, _Step


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

    @cached_property
    def trophic_levels(self) -> Dict[str, int]:
        """The trophic level of each step in the DAG.

        Returns:
            A dictionary with the trophic level of each step.
        """
        return {step: int(level) for step, level in nx.trophic_levels(self.G).items()}

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
        v = defaultdict(list)
        for step, trophic_level in self.trophic_levels.items():
            v[trophic_level].append(step)

        for trophic_level in sorted(v.keys()):
            yield v[trophic_level]

    def get_step_trophic_level(self, step_name: str) -> int:
        """Gets the trophic level of a step.

        Args:
            step_name: The name of the step.

        Returns:
            The trophic level of the step.
        """
        return int(self.trophic_levels[step_name])

    def is_step_in_trophic_level(self, step_name: str, trophic_level: int) -> bool:
        """Checks if a step is in a given trophic level.

        Args:
            step_name: The name of the step.
            trophic_level: The trophic level.

        Returns:
            True if the step is in the given trophic level, False otherwise.
        """
        return self.get_step_trophic_level(step_name) == trophic_level

    def step_in_last_trophic_level(self, step_name: str) -> bool:
        """Checks if a step is in the last trophic level.

        Args:
            step_name: The name of the step.

        Returns:
            True if the step is in the last trophic level, False otherwise.
        """
        return self.is_step_in_trophic_level(
            step_name, max(self.trophic_levels.values())
        )

    def validate(self) -> None:
        """Validates that the `Step`s included in the pipeline are correctly connected and
        have the correct inputs and outputs.

        Raises:
            ValueError: If the pipeline is not valid.
        """

        steps_receiving_routed_batches = []

        for trophic_level, steps in enumerate(
            self.iter_based_on_trophic_levels(), start=1
        ):
            for step_name in steps:
                node = self.get_step(step_name)
                step: "_Step" = node[STEP_ATTR_NAME]

                # Check if the step `process` function has `StepInput` argument
                self._validate_step_process_arguments(step)

                # Check if the required runtime parameters are provided
                self._validate_step_process_runtime_parameters(step)

                # Validate step mappings
                step.verify_inputs_mappings()
                step.verify_outputs_mappings()

                # Validate that the steps in the first trophic level are `GeneratorStep`s
                if trophic_level == 1:
                    if not isinstance(step, GeneratorStep):
                        raise ValueError(
                            f"Step '{step_name}' cannot be a root step because it is not"
                            " a `GeneratorStep`. It should have a previous step in the pipeline."
                        )
                    self._validate_generator_step_process_signature(step)
                else:
                    self._step_inputs_are_available(step)

                    # Validate routing batch function (if any)
                    predecessors = list(self.get_step_predecessors(step.name))  # type: ignore
                    self._validate_convergence_step(
                        step,
                        predecessors,
                        steps_receiving_routed_batches,  # type: ignore
                    )
                    receives_routed_batches = self._validate_routing_batch_function(
                        step, predecessors
                    )
                    if receives_routed_batches:
                        steps_receiving_routed_batches.append(step.name)

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
            for output in self.get_step(step_name)[STEP_ATTR_NAME].get_outputs()  # type: ignore
        ]
        step_inputs = step.get_inputs()
        if not all(input in inputs_available_for_step for input in step_inputs):
            raise ValueError(
                f"Step '{step.name}' requires inputs {step_inputs}, but only the inputs"
                f"={inputs_available_for_step} are available, which means that the inputs"
                f"={list(set(step_inputs) - set(inputs_available_for_step))} are missing or not"
                " available when the step gets to be executed in the pipeline."
                f" Please make sure previous steps to '{step.name}' are generating"
                " the required inputs."
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
        self._validate_process_step_input_parameter(step.name, step_input_parameter)  # type: ignore

    def _validate_convergence_step(
        self,
        step: "Step",
        predecessors: List[str],
        steps_receiving_routed_batches: List[str],
    ) -> None:
        """Checks if the `step` is a convergence step (receiving batches from steps to
        which the batches were routed). If so, it validates that all the predecessors of
        the steps receives routed batches from the same step, and that the `input_batch_size`
        of the `step` is equal or lower to the `input_batch_size` of the previous steps.

        Args:
            step: The step to validate.
            predecessors: The predecessors of the step.
            steps_receiving_routed_batches: The steps that are receiving routed batches
                from other steps in the pipeline.
        """
        if not any(
            predecessor in steps_receiving_routed_batches
            for predecessor in predecessors
        ):
            return

        # Mark the step as a convergence step
        self.set_step_attr(step.name, CONVERGENCE_STEP_ATTR_NAME, True)  # type: ignore

        # Check if all the predecessors of the step are receiving routed batches from the
        # same step
        previous_steps_predecessors = [
            list(self.get_step_predecessors(predecessor))
            for predecessor in predecessors
        ]
        if not all(
            prev_step_predecessors == previous_steps_predecessors[0]
            for prev_step_predecessors in previous_steps_predecessors
        ):
            raise ValueError(
                f"Convergence step '{step.name}' should receive batches from steps receiving"
                " routed batches from the same previous step and `routing_batch_function`."
            )

        # Check if the `input_batch_size` of the step is equal or lower than the
        for predecessor in predecessors:
            prev_step: "Step" = self.get_step(predecessor)[STEP_ATTR_NAME]
            if step.input_batch_size > prev_step.input_batch_size:  # type: ignore
                raise ValueError(
                    "A convergence step should have an `input_batch_size` equal or lower"
                    " than the `input_batch_size` of the connected previous steps."
                    f" Convergence step '{step.name}' has an `input_batch_size` of {step.input_batch_size}"
                    f" and the previous step '{prev_step.name}' has an `input_batch_size`"
                    f" of {prev_step.input_batch_size}."
                )

    def _validate_routing_batch_function(
        self, step: "_Step", predecessors: List[str]
    ) -> bool:
        """Checks if the `step` is going to receive routed batches (i.e. `routing_batch_function`
        chooses which batches from upstream step goes to the downstream step). If so, then it
        validates that the step has only one predecessor and that its `input_batch_size` is
        equal or lower than the `input_batch_size` or `batch_size` of the previous step.
        These are requirements to keep batches synchronized when executing the pipeline.

        Args:
            step: The step to validate.
            predecessors: The predecessors of the step.

        Returns:
            `True` if the `step` is going to receive routed batches, `False` otherwise.

        Raises:
            ValueError: If the `step` is going to receive routed batches and it has multiple
                predecessors or its `input_batch_size` is higher than the previous step
                `input_batch_size` or `batch_size`.
        """
        routing_batch_function = None
        for predecessor in predecessors:
            node = self.get_step(predecessor)
            routing_batch_function = node.get(ROUTING_BATCH_FUNCTION_ATTR_NAME)
            if routing_batch_function is not None and len(predecessors) > 1:
                raise ValueError(
                    f"Step '{step.name}' cannot have multiple predecessors when the batches"
                    " of one are being routed with a `routing_batch_function`."
                )

        if routing_batch_function is None:
            return False

        # If the step receives routed batches, then check its `input_batch_size` is lower
        # or equal to the `input_batch_size` or `batch_size` of the previous step from which
        # the batches are being routed.
        predecessor_step: "_Step" = self.get_step(predecessors[0])[STEP_ATTR_NAME]  # type: ignore
        batch_size = (
            predecessor_step.batch_size  # type: ignore
            if predecessor_step.is_generator
            else predecessor_step.input_batch_size  # type: ignore
        )
        if step.input_batch_size > batch_size:  # type: ignore
            raise ValueError(
                f"Step '{step.name}' should have an `input_batch_size` equal or lower"
                f" than the `input_batch_size` or `batch_size` of the previous step."
                f" This is because the batches are being routed with a `routing_batch_function`"
                f" from step '{predecessor_step.name}' to step '{step.name}'."
            )

        if batch_size % step.input_batch_size != 0:  # type: ignore
            raise ValueError(
                f"Step '{step.name}' should have an `input_batch_size` that is a multiple"
                f" of the `input_batch_size` or `batch_size` of the previous step."
                f" This is because the batches are being routed with a `routing_batch_function`"
                f" from step '{predecessor_step.name}' to step '{step.name}'."
            )

        return True

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
            step_name: self.get_step(step_name)[STEP_ATTR_NAME]
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

    def _validate_step_process_runtime_parameters(  # noqa: C901
        self, step: "_Step"
    ) -> None:
        """Validates that the required runtime parameters of the step are provided. A
        runtime parameter is considered required if it doesn't have a default value. The
        name of the runtime parameters are separated by dots to represent nested parameters.

        Args:
            step: The step to validate.

        Raises:
            ValueError: If not all the required runtime parameters haven't been provided
                with a value.
        """

        def _get_pipeline_aux_code(step_name: str, param_name: str) -> str:
            parts = param_name.split(".")
            result = f'pipeline.run(parameters={{"{step_name}":'
            nested_dict = "..."
            for part in reversed(parts):
                nested_dict = f' {{"{part}": {nested_dict}}}'
            result += nested_dict + "})"
            return result

        def _get_attribute_default(
            step: "_Step", composed_param_name: str
        ) -> Union[Any, None]:
            parts = composed_param_name.split(".")
            attr = step
            for part in parts:
                if isinstance(attr, dict):
                    attr = attr.get(part, None)
                elif isinstance(attr, object):
                    attr = getattr(attr, part)
            return attr

        def _check_required_parameter(
            param_name: str,
            composed_param_name: str,
            is_optional_or_nested: Union[bool, "RuntimeParametersNames"],
            runtime_parameters: Dict[str, Any],
            runtime_parameters_names: "RuntimeParametersNames",
        ) -> None:
            if isinstance(is_optional_or_nested, dict):
                runtime_parameters_names = runtime_parameters_names[param_name]  # type: ignore
                for subparam, value in runtime_parameters_names.items():
                    _check_required_parameter(
                        param_name=subparam,
                        composed_param_name=f"{composed_param_name}.{subparam}",
                        is_optional_or_nested=value,
                        # NOTE: `runtime_parameters` get is for the specific case of `LLM` in `Task`
                        runtime_parameters=runtime_parameters.get(
                            param_name, runtime_parameters
                        ),
                        runtime_parameters_names=runtime_parameters_names,
                    )
                return

            if (
                not is_optional_or_nested
                and param_name not in runtime_parameters
                and _get_attribute_default(
                    step=step, composed_param_name=composed_param_name
                )
                is None
            ):
                aux_code = _get_pipeline_aux_code(step.name, composed_param_name)
                raise ValueError(
                    f"Step '{step.name}' is missing required runtime parameter '{param_name}'."
                    " Please, provide a value for it when calling `Pipeline.run` method:\n\n"
                    f"    {aux_code}"
                )

        runtime_parameters_names = step.runtime_parameters_names
        for param_name, value in runtime_parameters_names.items():
            _check_required_parameter(
                param_name=param_name,
                composed_param_name=param_name,
                is_optional_or_nested=value,
                runtime_parameters=step._runtime_parameters,
                runtime_parameters_names=runtime_parameters_names,
            )

    def _validate_generator_step_process_signature(self, step: "GeneratorStep") -> None:
        """Validates that the `process` method of the `GeneratorStep` does not expect the
        `inputs` arg within the method signature, and also the `offset` arg should always
        be present.

        Args:
            step: The step to validate.

        Raises:
            ValueError: If the `process` method of the `GeneratorStep` expects the `inputs` arg.
            ValueError: If the `process` method of the `GeneratorStep` does not expect the `offset` arg.
        """
        if step.get_process_step_input() is not None:
            raise ValueError(
                f"Generator step '{step.name}' should not have a parameter with type hint"
                " `StepInput` within the `process` method signature."
            )
        if not any("offset" == parameter.name for parameter in step.process_parameters):
            raise ValueError(
                f"Generator step '{step.name}' should have an `offset` parameter within"
                " the `process` method signature."
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

        data = {"steps": [], "connections": [], "routing_batch_functions": []}
        for i, node in enumerate(adjacency_data["nodes"]):
            name = node["id"]
            data["steps"].append({"step": node[STEP_ATTR_NAME].dump(), "name": name})
            data["connections"].append(
                {
                    "from": name,
                    "to": [node["id"] for node in adjacency_data["adjacency"][i]],
                }
            )
            if routing_batch_function := node.get(ROUTING_BATCH_FUNCTION_ATTR_NAME):
                data["routing_batch_functions"].append(routing_batch_function.dump())

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
            cls_step: Type["_Step"] = _get_module_attr(
                **step[STEP_ATTR_NAME][TYPE_INFO_KEY]
            )
            dag.add_step(cls_step.from_dict(step[STEP_ATTR_NAME]))

        for connection in data["connections"]:
            from_step = connection["from"]
            for to_step in connection["to"]:
                dag.add_edge(from_step, to_step)

        for routing_batch_function in data.get("routing_batch_functions", []):
            step_name = routing_batch_function["step"]
            routing_batch_function = RoutingBatchFunction.from_dict(
                routing_batch_function
            )
            dag.set_step_attr(
                name=step_name,
                attr=ROUTING_BATCH_FUNCTION_ATTR_NAME,
                value=routing_batch_function,
            )

        return dag
