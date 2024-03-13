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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import PrivateAttr
from vllm import LLM as _vLLM
from vllm import SamplingParams

from distilabel.llm.base import LLM

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from distilabel.steps.task.typing import ChatType


class vLLM(LLM):
    """To run `vLLM` the following environment variable needs to be set in advance:
    `OPENBLAS_NUM_THREADS=1`.
    """

    model: str
    model_kwargs: Optional[Dict[str, Any]] = {}
    chat_format: Optional[str] = None

    _model: Optional["_vLLM"] = PrivateAttr(...)
    _tokenizer: Optional["PreTrainedTokenizer"] = PrivateAttr(...)

    def load(self) -> None:
        self._model = _vLLM(self.model, **self.model_kwargs)  # type: ignore
        self._tokenizer = self._model.get_tokenizer()  # type: ignore

    @property
    def model_name(self) -> str:
        return self.model

    def prepare_input(self, input: "ChatType") -> str:
        return self._tokenizer.apply_chat_template(  # type: ignore
            input,  # type: ignore
            tokenize=False,
            add_generation_prompt=True,  # type: ignore
        )

    # TODO: update to return `List[List[str]]` depending on the `num_generations` parameter
    def generate(
        self,
        inputs: List["ChatType"],
        num_generations: int = 1,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        **extra_sampling_params: Any,
    ) -> List[List[Union[str, None]]]:
        sampling_params = SamplingParams(  # type: ignore
            n=num_generations,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            **extra_sampling_params,
        )

        prepared_inputs = [self.prepare_input(input) for input in inputs]
        batch_outputs = self._model.generate(  # type: ignore
            prepared_inputs,
            sampling_params,
            use_tqdm=False,  # type: ignore
        )
        return [[output.text for output in outputs] for outputs in batch_outputs]
