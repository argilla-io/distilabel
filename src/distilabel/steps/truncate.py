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
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from typing_extensions import override

from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class TruncateTextColumn(Step):
    """Truncate a row using a tokenizer or the number of characters.

    `TruncateTextColumn` is a `Step` that truncates a row according to the max length. If
    the `tokenizer` is provided, then the row will be truncated using the tokenizer,
    and the `max_length` will be used as the maximum number of tokens, otherwise it will
    be used as the maximum number of characters. The `TruncateTextColumn` step is useful when one
    wants to truncate a row to a certain length, to avoid posterior errors in the model due
    to the length.

    Attributes:
        column: the column to truncate. Defaults to `"text"`.
        max_length: the maximum length to use for truncation.
            If a `tokenizer` is given, corresponds to the number of tokens,
            otherwise corresponds to the number of characters. Defaults to `8192`.
        tokenizer: the name of the tokenizer to use. If provided, the row will be
            truncated using the tokenizer. Defaults to `None`.

    Input columns:
        - dynamic (determined by `column` attribute): The columns to be truncated, defaults to "text".

    Output columns:
        - dynamic (determined by `column` attribute): The truncated column.

    Categories:
        - text-manipulation

    Examples:
        Truncating a row to a given number of tokens:

        ```python
        from distilabel.steps import TruncateTextColumn

        trunc = TruncateTextColumn(
            tokenizer="meta-llama/Meta-Llama-3.1-70B-Instruct",
            max_length=4,
            column="text"
        )

        trunc.load()

        result = next(
            trunc.process(
                [
                    {"text": "This is a sample text that is longer than 10 characters"}
                ]
            )
        )
        # result
        # [{'text': 'This is a sample'}]
        ```

        Truncating a row to a given number of characters:

        ```python
        from distilabel.steps import TruncateTextColumn

        trunc = TruncateTextColumn(max_length=10)

        trunc.load()

        result = next(
            trunc.process(
                [
                    {"text": "This is a sample text that is longer than 10 characters"}
                ]
            )
        )
        # result
        # [{'text': 'This is a '}]
        ```
    """

    column: str = "text"
    max_length: int = 8192
    tokenizer: Optional[str] = None
    _truncator: Optional[Callable[[str], str]] = None
    _tokenizer: Optional[Any] = None

    def load(self):
        super().load()
        if self.tokenizer:
            if not importlib.util.find_spec("transformers"):
                raise ImportError(
                    "`transformers` is needed to tokenize, but is not installed. "
                    "Please install it using `pip install transformers`."
                )

            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
            self._truncator = self._truncate_with_tokenizer
        else:
            self._truncator = self._truncate_with_length

    @property
    def inputs(self) -> List[str]:
        return [self.column]

    @property
    def outputs(self) -> List[str]:
        return self.inputs

    def _truncate_with_length(self, text: str) -> str:
        """Truncates the text according to the number of characters."""
        return text[: self.max_length]

    def _truncate_with_tokenizer(self, text: str) -> str:
        """Truncates the text according to the number of characters using the tokenizer."""
        return self._tokenizer.decode(
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=True,
            )
        )

    @override
    def process(self, inputs: StepInput) -> "StepOutput":
        for input in inputs:
            input[self.column] = self._truncator(input[self.column])
        yield inputs
