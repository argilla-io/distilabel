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

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import PrivateAttr
from transformers import Pipeline, pipeline

from distilabel.pipeline.llm.base import LLM

if TYPE_CHECKING:
    from distilabel.pipeline.step.task.typing import ChatType


class TransformersLLM(LLM):
    model: str
    revision: str = "main"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    model_kwargs: Optional[Dict[str, Any]] = None
    tokenizer: Optional[str] = None
    use_fast: bool = True
    chat_template: Optional[str] = None
    device: Optional[Union[str, int]] = None
    device_map: Optional[Union[str, Dict[str, Any]]] = None
    token: Optional[str] = None

    _pipeline: Optional["Pipeline"] = PrivateAttr(...)

    def load(self) -> None:
        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=self.model_kwargs or {},
            tokenizer=self.tokenizer or self.model,
            use_fast=self.use_fast,
            device=self.device,
            device_map=self.device_map,
            token=self.token or os.getenv("HF_TOKEN"),
            return_full_text=False,
        )

        if self.chat_template is not None:
            self._pipeline.tokenizer.chat_template = self.chat_template  # type: ignore
        elif (
            self._pipeline.tokenizer.chat_template is None  # type: ignore
            and self._pipeline.tokenizer.default_chat_template is None  # type: ignore
        ):
            self._pipeline.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"  # type: ignore

    @property
    def model_name(self) -> str:
        return self.model

    def prepare_input(self, input: "ChatType") -> str:
        return self._pipeline.tokenizer.apply_chat_template(  # type: ignore
            input,  # type: ignore
            tokenize=False,
            add_generation_prompt=True,  # type: ignore
        )

    def generate(
        self,
        inputs: List["ChatType"],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
    ) -> List[str]:
        outputs = []
        for input in inputs:
            output = self._pipeline(  # type: ignore
                self.prepare_input(input=input),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
            outputs.append(output[0]["generated_text"])  # type: ignore
        return outputs
