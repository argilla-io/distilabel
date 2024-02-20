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

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing_extensions import Annotated

from distilabel.pipeline.base import BasePipeline, _GlobalPipelineManager
from distilabel.utils.serialization_v2 import _get_class, _Serializable

StepInput = Annotated[List[Dict[str, Any]], "StepInput"]
"""StepInput is just an `Annotated` alias of the typing `List[Dict[str, Any]]` with
extra metadata that allows `distilabel` to perform validations over the `process` step
method defined in each `Step`"""


class Step(BaseModel, _Serializable, ABC):
    """Base class for the steps that can be included in a `Pipeline`.

    A `Step` is a class defining some processing logic. The input and outputs for this
    processing logic are lists of dictionaries with the same keys:

        ```python
        [
            {"column1": "value1", "column2": "value2", ...},
            {"column1": "value1", "column2": "value2", ...},
            {"column1": "value1", "column2": "value2", ...},
        ]
        ```

    The processing logic is defined in the `process` method, which depending on the
    number of previous steps, can receive more than one list of dictionaries, each with
    the output of the previous steps. In order to make `distilabel` know where the outputs
    from the previous steps are, the `process` function from each `Step` must have an argument
    or positional argument annotated with `StepInput`. Additionally, the `process` method
    can have any number of arguments, which will be the runtime arguments of the step.

        ```python
        class StepWithOnePreviousStep(Step):
            def process(self, input: StepInput, runtime_argument1: str, runtime_argument2: int) -> List[Dict[str, Any]]:
                pass

        class StepWithSeveralPreviousStep(Step):
            # mind the * to indicate that the argument is a list of StepInput
            def process(self, input: *StepInput) -> List[Dict[str, Any]]:
                pass
        ```

    In order to perform static validations and to check that the chaining of the steps
    in the pipeline is valid, a `Step` must also define the `inputs` and `outputs`
    properties:

    - `inputs`: a list of strings with the names of the columns that the step needs as
        input. It can be an empty list if the step is a generator step.
    - `outputs`: a list of strings with the names of the columns that the step will
        produce as output.

    Optionally, a `Step` can override the `load` method to perform any initialization
    logic before the `process` method is called. For example, to load an LLM, stablish a
    connection to a database, etc.

    Finally, the `Step` class inherits from `pydantic.BaseModel`, so attributes can be easily
    defined, validated, serialized and included in the `__init__` method of the step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    pipeline: Annotated[Union[BasePipeline, None], Field(exclude=True)] = None

    _runtime_parameters: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _: Any) -> None:
        if self.pipeline is None:
            self.pipeline = _GlobalPipelineManager.get_pipeline()

        if self.pipeline is None:
            raise ValueError(
                f"Step '{self.name}' hasn't received a pipeline, and it hasn't been"
                " created within a `Pipeline` context. Please, use"
                " `with Pipeline() as pipeline:` and create the step within the context."
            )

        self.pipeline._add_step(self)

    def connect(
        self, step: "Step", input_mapping: Union[Dict[str, Any], None] = None
    ) -> None:
        """Connects the current step to another step in the pipeline, which means that
        the output of this step will be the input of the other step.

        Args:
            step: The step to connect to.
            input_mapping: A dictionary with the mapping of the columns from the output
                of the current step to the input of the other step. If `None`, the
                columns will be mapped by name. This is useful when the names of the
                output columns of the current step are different from the names of the
                input columns of the other step. Defaults to `None`.
        """

        self.pipeline._add_edge(self.name, step.name)  # type: ignore

    def load(self) -> None:
        """Method to perform any initialization logic before the `process` method is
        called. For example, to load an LLM, stablish a connection to a database, etc.
        """
        pass

    def _set_runtime_parameters(self, runtime_parameters: Dict[str, Any]) -> None:
        """Sets the runtime parameters of the step.

        Args:
            runtime_parameters: A dictionary with the runtime parameters for the step.
        """
        self._runtime_parameters = runtime_parameters

    @property
    def is_generator(self) -> bool:
        """Whether the step is a generator step or not.

        Returns:
            `True` if the step is a generator step, `False` otherwise.
        """
        return isinstance(self, GeneratorStep)

    @property
    def is_global(self) -> bool:
        """Whether the step is a global step or not.

        Returns:
            `True` if the step is a global step, `False` otherwise.
        """
        return isinstance(self, GlobalStep)

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        """List of strings with the names of the columns that the step needs as input.

        Returns:
            List of strings with the names of the columns that the step needs as input.
        """
        pass

    @property
    def outputs(self) -> List[str]:
        """List of strings with the names of the columns that the step will produce as
        output.

        Returns:
            List of strings with the names of the columns that the step will produce as
            output.
        """
        return []

    @abstractmethod
    def process(
        self, *args: Any, **kwargs: Any
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Method that defines the processing logic of the step."""
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        """Create a Step from a dict containing the serialized data.

        Needs the information from the step and the Pipeline it belongs to.

        Note:
            It's intended for internal use.

        Args:
            data (Dict[str, Any]): Dict containing the serialized data from a Step and the
                Pipeline it belongs to.

        Returns:
            step (Step): Instance of the Step.
        """

        name = data.pop("name")
        type_info = data.pop("_type_info_")
        cls_pipeline = _get_class(type_info["module"], type_info["name"])
        # By this moment, the data should only contain the dag info,
        # we need to create the generic type of the pipeline that is contained
        # in the data dictionary.
        if pipe := _GlobalPipelineManager.get_pipeline():
            pipeline = pipe
        else:
            pipeline = cls_pipeline.from_dict(data)

        step = cls(name=name, pipeline=pipeline)
        # NOTE(plaguss): Still needs to check for the inputs/outputs of higher order classes.
        return step


class GeneratorStep(Step, ABC):
    """A special kind of `Step` that is able to generate data i.e. it doesn't receive
    any input from the previous steps.
    """

    @property
    def inputs(self) -> List[str]:
        return []


class GlobalStep(Step, ABC):
    """A special kind of `Step` which it's `process` method receives all the data processed
    by their previous steps at once, instead of receiving it in batches. This kind of step
    are useful when the processing logic requires to have all the data at once, for example
    to train a model, to perform a global aggregation, etc.
    """

    pass
