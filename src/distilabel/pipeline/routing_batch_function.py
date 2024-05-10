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
import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from pydantic import BaseModel, PrivateAttr
from typing_extensions import Self

from distilabel.utils.serialization import (
    TYPE_INFO_KEY,
    _get_module_attr,
    _Serializable,
)

if TYPE_CHECKING:
    from distilabel.pipeline.base import _Batch
    from distilabel.pipeline.typing import DownstreamConnectableSteps
    from distilabel.steps.base import _Step

RoutingBatchFunc = Callable[[List[str]], List[str]]
"""Type alias for a routing batch function. It takes a list of all the downstream steps and
returns a list with the names of the steps that should receive the batch."""


class RoutingBatchFunction(BaseModel, _Serializable):
    """A thin wrapper around a routing batch function that can be used to route batches
    from one upstream step to specific downstream steps.

    Attributes:
        routing_function: The routing function that takes a list of all the downstream steps
            and returns a list with the names of the steps that should receive the batch.
        _step: The upstream step that is connected to the routing batch function.
        _routed_batch_registry: A dictionary that keeps track of the batches that have been
            routed to specific downstream steps.
    """

    routing_function: RoutingBatchFunc

    _step: Union["_Step", None] = PrivateAttr(default=None)
    _routed_batch_registry: Dict[str, Dict[int, List[str]]] = PrivateAttr(
        default_factory=dict
    )
    _factory_function_module: Union[str, None] = PrivateAttr(default=None)
    _factory_function_name: Union[str, None] = PrivateAttr(default=None)
    _factory_function_kwargs: Union[Dict[str, Any], None] = PrivateAttr(default=None)

    def route_batch(self, batch: "_Batch", steps: List[str]) -> List[str]:
        """Returns a list of selected downstream steps from `steps` to which the `batch`
        should be routed.

        Args:
            batch: The batch that should be routed.
            steps: A list of all the downstream steps that can receive the batch.

        Returns:
            A list with the names of the steps that should receive the batch.
        """
        routed_steps = self.routing_function(steps)
        self._register_routed_batch(batch, routed_steps)
        return routed_steps

    def set_factory_function(
        self,
        factory_function_module: str,
        factory_function_name: str,
        factory_function_kwargs: Dict[str, Any],
    ) -> None:
        """Sets the factory function that was used to create the `routing_batch_function`.

        Args:
            factory_function_module: The module name where the factory function is defined.
            factory_function_name: The name of the factory function that was used to create
                the `routing_batch_function`.
            factory_function_kwargs: The keyword arguments that were used when calling the
                factory function.
        """
        self._factory_function_module = factory_function_module
        self._factory_function_name = factory_function_name
        self._factory_function_kwargs = factory_function_kwargs

    def __call__(self, batch: "_Batch", steps: List[str]) -> List[str]:
        """Returns a list of selected downstream steps from `steps` to which the `batch`
        should be routed.

        Args:
            batch: The batch that should be routed.
            steps: A list of all the downstream steps that can receive the batch.

        Returns:
            A list with the names of the steps that should receive the batch.
        """
        return self.route_batch(batch, steps)

    def _register_routed_batch(self, batch: "_Batch", routed_steps: List[str]) -> None:
        """Registers a batch that has been routed to specific downstream steps.

        Args:
            batch: The batch that has been routed.
            routed_steps: The list of downstream steps that have been selected to receive
                the batch.
        """
        upstream_step = batch.step_name
        batch_seq_no = batch.seq_no
        self._routed_batch_registry.setdefault(upstream_step, {}).setdefault(
            batch_seq_no, routed_steps
        )

    def __rshift__(
        self, other: List["DownstreamConnectableSteps"]
    ) -> List["DownstreamConnectableSteps"]:
        """Connects a list of dowstream steps to the upstream step of the routing batch
        function.

        Args:
            other: A list of downstream steps that should be connected to the upstream step
                of the routing batch function.

        Returns:
            The list of downstream steps that have been connected to the upstream step of the
            routing batch function.
        """
        if not isinstance(other, list):
            raise ValueError(
                f"Can only set a `routing_batch_function` for a list of steps. Got: {other}."
                " Please, review the right-hand side of the `routing_batch_function >> other`"
                " expression. It should be"
                " `upstream_step >> routing_batch_function >> [downstream_step_1, dowstream_step_2, ...]`."
            )

        if not self._step:
            raise ValueError(
                "Routing batch function doesn't have an upstream step. Cannot connect downstream"
                " steps before connecting the upstream step. Connect this routing batch"
                " function to an upstream step using the `>>` operator. For example:"
                " `upstream_step >> routing_batch_function >> [downstream_step_1, downstream_step_2, ...]`."
            )

        for step in other:
            self._step.connect(step)
        return other

    def dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Dumps the routing batch function to a dictionary, and the information of the
        factory function used to create this routing batch function.

        Args:
            **kwargs: Additional keyword arguments that should be included in the dump.

        Returns:
            A dictionary with the routing batch function information and the factory function
            information.
        """
        dump_info: Dict[str, Any] = {"step": self._step.name}  # type: ignore

        if (
            self._factory_function_module
            and self._factory_function_name
            and self._factory_function_kwargs
        ):
            dump_info[TYPE_INFO_KEY] = {
                "module": self._factory_function_module,
                "name": self._factory_function_name,
                "kwargs": self._factory_function_kwargs,
            }

        return dump_info

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Loads a routing batch function from a dictionary. It must contain the information
        of the factory function used to create the routing batch function.

        Args:
            data: A dictionary with the routing batch function information and the factory
                function information.
        """
        type_info = data.get(TYPE_INFO_KEY)
        if not type_info:
            step = data.get("step")
            raise ValueError(
                f"The routing batch function for step '{step}' was created without a factory"
                " function, and it cannot be reconstructed."
            )

        module = type_info.get("module")
        name = type_info.get("name")
        kwargs = type_info.get("kwargs")

        if not module or not name or not kwargs:
            raise ValueError(
                "The routing batch function was created with a factory function, but the"
                " information is incomplete. Cannot reconstruct the routing batch function."
            )

        routing_batch_function = _get_module_attr(module=module, name=name)(**kwargs)
        routing_batch_function.set_factory_function(
            factory_function_module=module,
            factory_function_name=name,
            factory_function_kwargs=kwargs,
        )

        return routing_batch_function


def routing_batch_function(func: RoutingBatchFunc) -> RoutingBatchFunction:
    """Creates a routing batch function that can be used to route batches from one upstream
    step to specific downstream steps.

    Args:
        func: The routing function that takes a list of all the downstream steps and returns
            a list with the names of the steps that should receive the batch.

    Returns:
        A `RoutingBatchFunction` instance that can be used with the `>>` operators and with
        the `Pipeline.connect` method when defining the pipeline.

    Example:

    ```python
    from distilabel.llms import MistralLLM, OpenAILLM, VertexAILLM
    from distilabel.pipeline import Pipeline, routing_batch_function
    from distilabel.steps import LoadHubDataset, CombineColumns


    @routing_batch_function
    def random_routing_batch(steps: List[str]) -> List[str]:
        return random.sample(steps, 2)


    with Pipeline(name="routing-batch-function") as pipeline:
        load_data = LoadHubDataset()

        generations = []
        for llm in (
            OpenAILLM(model="gpt-4-0125-preview"),
            MistralLLM(model="mistral-large-2402"),
            VertexAILLM(model="gemini-1.5-pro"),
        ):
            task = TextGeneration(name=f"text_generation_with_{llm.model_name}", llm=llm)
            generations.append(task)

        combine_columns = CombineColumns(columns=["generation", "model_name"])

        load_data >> random_routing_batch >> generations >> combine_columns
    ```
    """

    factory_function_name, factory_function_module, factory_function_kwargs = (
        None,
        None,
        None,
    )

    # Check if `routing_batch_function` was created using a factory function from an installed package
    stack = inspect.stack()
    if len(stack) > 2:
        factory_function_frame_info = stack[1]

        # Function factory path
        factory_function_name = factory_function_frame_info.function
        factory_function_module = inspect.getmodule(
            factory_function_frame_info.frame
        ).__name__  # type: ignore

        # Function factory kwargs
        factory_function_kwargs = factory_function_frame_info.frame.f_locals

    routing_batch_function = RoutingBatchFunction(routing_function=func)

    if factory_function_module and factory_function_name and factory_function_kwargs:
        routing_batch_function.set_factory_function(
            factory_function_module=factory_function_module,
            factory_function_name=factory_function_name,
            factory_function_kwargs=factory_function_kwargs,
        )

    return routing_batch_function


def sample_n_steps(n: int) -> RoutingBatchFunction:
    """A simple function that creates a routing batch function that samples `n` steps from
    the list of all the downstream steps.

    Args:
        n: The number of steps to sample from the list of all the downstream steps.

    Returns:
        A `RoutingBatchFunction` instance that can be used with the `>>` operators and with
        the `Pipeline.connect` method when defining the pipeline.

    Example:

    ```python
    from distilabel.llms import MistralLLM, OpenAILLM, VertexAILLM
    from distilabel.pipeline import Pipeline, sample_n_steps
    from distilabel.steps import LoadHubDataset, CombineColumns


    random_routing_batch = sample_n_steps(2)


    with Pipeline(name="routing-batch-function") as pipeline:
        load_data = LoadHubDataset()

        generations = []
        for llm in (
            OpenAILLM(model="gpt-4-0125-preview"),
            MistralLLM(model="mistral-large-2402"),
            VertexAILLM(model="gemini-1.5-pro"),
        ):
            task = TextGeneration(name=f"text_generation_with_{llm.model_name}", llm=llm)
            generations.append(task)

        combine_columns = CombineColumns(columns=["generation", "model_name"])

        load_data >> random_routing_batch >> generations >> combine_columns
    ```
    """

    @routing_batch_function
    def sample_n(steps: List[str]) -> List[str]:
        return random.sample(steps, n)

    return sample_n
