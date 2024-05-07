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

from typing import TYPE_CHECKING, Callable, Dict, List, Union

from pydantic import BaseModel, PrivateAttr

from distilabel.utils.serialization import _Serializable

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
    """

    routing_function: RoutingBatchFunc

    _step: Union["_Step", None] = PrivateAttr(default=None)
    _routed_batch_registry: Dict[str, Dict[int, List[str]]] = PrivateAttr(
        default_factory=dict
    )

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
    return RoutingBatchFunction(routing_function=func)
