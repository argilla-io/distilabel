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
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Union,
    get_args,
    get_origin,
)

import networkx as nx
from typing_extensions import Annotated

if TYPE_CHECKING:
    from distilabel.step.base import Step


def _is_step_input(parameter: inspect.Parameter) -> bool:
    return (
        get_origin(parameter.annotation) is Annotated
        and get_args(parameter.annotation)[-1] == "StepInput"
    )


class DAG:
    def __init__(self) -> None:
        self.dag = nx.DiGraph()

    def add_step(self, step: "Step") -> None:
        name = step.name
        if name in self.dag:
            raise ValueError(f"Step with name '{name}' already exists")
        self.dag.add_node(name, step=step)

    def get_step(self, name: str) -> "Step":
        if name not in self.dag:
            raise ValueError(f"Step with name '{name}' does not exist")
        return self.dag.nodes[name]["step"]

    def add_edge(self, from_step: str, to_step: str) -> None:
        if from_step not in self.dag:
            raise ValueError(f"Step with name '{from_step}' does not exist")

        if to_step not in self.dag:
            raise ValueError(f"Step with name '{to_step}' does not exist")

        if to_step in self.dag[from_step]:
            raise ValueError(
                f"There is already a edge from '{to_step}' to '{from_step}'"
            )

        if to_step in nx.ancestors(self.dag, from_step):
            raise ValueError(
                f"Cannot add edge from '{from_step}' to '{to_step}' as it would create a cycle."
            )

        self.dag.add_edge(from_step, to_step)

    def iter_based_on_trophic_levels(self) -> Iterable[List[str]]:
        """Iterate over steps names in the DAG based on their trophic levels. This is similar
        to a topological sort, but we also know which steps are at the same level and
        can be run in parallel.

        Yields:
            A list containing the names of the steps that can be run in parallel.
        """
        trophic_levels = nx.trophic_levels(self.dag)

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
                step = self.get_step(step_name)

                # Validate that the steps in the first trophic level are `GeneratorStep`s
                if trophic_level == 1:
                    if not step.is_generator:
                        raise ValueError(
                            f"Step '{step_name}' should be `GeneratorStep` as it doesn't have any previous steps"
                        )
                else:
                    self._step_inputs_are_available(step)

                self._validate_step_process_arguments(step)

    def _step_inputs_are_available(self, step: "Step") -> None:
        """Validates that the `Step.inputs` will be available when the step gets to be
        executed in the pipeline i.e. the step will receive list of dictionaries containing
        its inputs as keys.

        Args:
            step: The step.
        """
        inputs_available_for_step = [
            output
            for step_name in nx.ancestors(self.dag, step.name)
            for output in self.get_step(step_name).outputs
        ]
        if not all(input in inputs_available_for_step for input in step.inputs):
            step_inputs = ", ".join([f"'{input}'" for input in step.inputs])
            raise ValueError(
                f"Step '{step.name}' requires inputs {step_inputs} which are not"
                f" available when the step gets to be executed in the pipeline."
                f" Please make sure previous steps to '{step.name}' are generating"
                f" the required inputs."
            )

    def _validate_step_process_arguments(self, step: "Step") -> None:
        """Validates the arguments of the `Step.process` method, checking there is an
        argument with type hint `StepInput` and that all the required runtime parameters
        are provided.

        Args:
            step: The step to validate.

        Raises:
            ValueError: If the arguments of the `process` method of the step are not valid.
        """
        signature = inspect.signature(step.process)

        step_input_parameter = None
        runtime_parameters = []

        for parameter in signature.parameters.values():
            if _is_step_input(parameter):
                if step_input_parameter is not None:
                    raise ValueError(
                        f"Step '{step.name}' should have only one parameter with type hint `StepInput`."
                    )
                step_input_parameter = parameter
            else:
                runtime_parameters.append(parameter)

        self._validate_process_step_input_parameter(step.name, step_input_parameter)
        self._validate_step_process_runtime_parameters(step, runtime_parameters)

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
            step_name: self.get_step(step_name)
            for step_name in self.dag.predecessors(step_name)
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

    def _validate_step_process_runtime_parameters(
        self, step: "Step", parameters: List[inspect.Parameter]
    ) -> None:
        """Validates that the required runtime parameters of the step are provided.

        Args:
            step: The step to validate.
            parameters: The runtime parameters of the `Step.process` method.

        Raises:
            ValueError: If not all the required runtime parameters haven't been provided
                with a value.
        """
        runtime_parameters_values = step._runtime_parameters
        for parameter in parameters:
            if (
                parameter.default == inspect.Parameter.empty
                and parameter.name not in runtime_parameters_values
            ):
                raise ValueError(
                    f"Step '{step.name}' is missing required runtime parameter '{parameter.name}'."
                    " Please, provide a value for it when calling `Pipeline.run`"
                )
