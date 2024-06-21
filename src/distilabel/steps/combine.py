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

from typing import TYPE_CHECKING, List, Optional

from typing_extensions import override

from distilabel.pipeline.utils import combine_dicts, combine_keys
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class CombineColumns(Step):
    """Combines columns from a list of `StepInput`.

    `CombineColumns` is a `Step` that implements the `process` method that calls the `combine_dicts`
    function to handle and combine a list of `StepInput`. Also `CombineColumns` provides two attributes
    `columns` and `output_columns` to specify the columns to merge and the output columns
    which will override the default value for the properties `inputs` and `outputs`, respectively.

    Attributes:
        columns: List of strings with the names of the columns to merge.
        output_columns: Optional list of strings with the names of the output columns.

    Input columns:
        - dynamic (determined by `columns` attribute): The columns to merge.

    Output columns:
        - dynamic (determined by `columns` and `output_columns` attributes): The columns
            that were merged.

    Examples:

        Combine columns of a dataset:

        ```python
        from distilabel.steps import CombineColumns

        combine_columns = CombineColumns(
            name="combine_columns",
            columns=["generation", "model_name"],
        )
        combine_columns.load()

        result = next(
            combine_columns.process(
                [{"generation": "AI generated text"}, {"model_name": "my_model"}],
                [{"generation": "Other generated text", "model_name": "my_model"}]
            )
        )
        # >>> result
        # [{'merged_generation': ['AI generated text', 'Other generated text'], 'merged_model_name': ['my_model']}]
        ```

        Specify the name of the output columns:

        ```python
        from distilabel.steps import CombineColumns

        combine_columns = CombineColumns(
            name="combine_columns",
            columns=["generation", "model_name"],
            output_columns=["generations", "generation_models"]
        )
        combine_columns.load()

        result = next(
            combine_columns.process(
                [{"generation": "AI generated text"}, {"model_name": "my_model"}],
                [{"generation": "Other generated text", "model_name": "my_model"}]
            )
        )
        # >>> result
        #[{'generations': ['AI generated text', 'Other generated text'], 'generation_models': ['my_model']}]
        ```
    """

    columns: List[str]
    output_columns: Optional[List[str]] = None

    @property
    def inputs(self) -> List[str]:
        """The inputs for the task are the column names in `columns`."""
        return self.columns

    @property
    def outputs(self) -> List[str]:
        """The outputs for the task are the column names in `output_columns` or
        `merged_{column}` for each column in `columns`."""
        return (
            self.output_columns
            if self.output_columns is not None
            else [f"merged_{column}" for column in self.columns]
        )

    @override
    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method calls the `combine_dicts` function to handle and combine a list of `StepInput`.

        Args:
            *inputs: A list of `StepInput` to be combined.

        Yields:
            A `StepOutput` with the combined `StepInput` using the `combine_dicts` function.
        """
        yield combine_dicts(
            *inputs,
            merge_keys=self.inputs,
            output_merge_keys=self.outputs,
        )


class CombineKeys(Step):
    """Combines keys from a row.

    `CombineKeys` is a `Step` that implements the `process` method that calls the `combine_keys`
    function to handle and combine keys in a `StepInput`. `CombineKeys` provides two attributes
    `keys` and `output_keys` to specify the keys to merge and the resulting output key.

    This step can be useful if you have a `Task` that generates instructions for example, and you
    want to have more examples of those. In such a case, you could for example use another `Task`
    to multiply your instructions synthetically, what would yield two different keys splitted.
    Using `CombineKeys` you can merge them and use them as a single column in your dataset for
    further processing.

    Attributes:
        columns: List of strings with the names of the columns to merge.
        output_columns: Optional list of strings with the names of the output columns.

    Input columns:
        - dynamic (determined by `keys` attribute): The keys to merge.

    Output columns:
        - dynamic (determined by `keys` and `output_key` attributes): The columns
            that were merged.

    Examples:

        Combine keys in rows of a dataset:

        ```python
        from distilabel.steps import CombineKeys

        combiner = CombineKeys(
            keys=["queries", "multiple_queries"],
            output_key="queries",
        )
        combiner.load()

        result = next(
            combiner.process(
                [
                    {
                        "queries": "How are you?",
                        "multiple_queries": ["What's up?", "Everything ok?"]
                    }
                ],
            )
        )
        # >>> result
        # [{'queries': ['How are you?', "What's up?", 'Everything ok?']}]
        ```
    """

    keys: List[str]
    output_key: Optional[str] = None

    @property
    def inputs(self) -> List[str]:
        return self.keys

    @property
    def outputs(self) -> List[str]:
        return [self.output_key] if self.output_key else ["combined_key"]

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        combined = []
        for input in inputs:
            combined.append(
                combine_keys(
                    input,
                    keys=self.keys,
                    new_key=self.outputs[0],
                )
            )
        yield combined
