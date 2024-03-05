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

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from llama_cpp import Llama
from pydantic import PrivateAttr

from distilabel.llm.base import LLM

if TYPE_CHECKING:
    from distilabel.pipeline.step.task.typing import ChatType


class LlamaCppLLM(LLM):
    model_path: Path
    chat_format: str = "chatml"
    n_gpu_layers: int = -1
    verbose: bool = False

    _model: Optional["Llama"] = PrivateAttr(...)

    def load(self) -> None:
        self._model = Llama(
            model_path=self.model_path.as_posix(),
            chat_format=self.chat_format,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )

    @property
    def model_name(self) -> str:
        return self._model.model_path  # type: ignore

    def generate(
        self,
        inputs: List["ChatType"],
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> List[str]:
        outputs = []
        for input in inputs:
            chat_completions = self._model.create_chat_completion(  # type: ignore
                messages=input,  # type: ignore
                max_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
            )
            outputs.append(
                chat_completions["choices"][0]["message"]["content"]  # type: ignore
            )
        return outputs
