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

import asyncio
import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Union, cast

from pydantic import field_validator

from distilabel.llms.base import LLM, AsyncLLM
from distilabel.steps.tasks.typing import StandardInput

if TYPE_CHECKING:
    from distilabel.llms.typing import GenerateOutput
    from distilabel.mixins.runtime_parameters import RuntimeParametersNames
    from distilabel.steps.tasks.typing import FormattedInput

MOA_SYSTEM_PROMPT = (
    "You have been provided with a set of responses from various open-source models to the"
    " latest user query. Your task is to synthesize these responses into a single, high-quality"
    " response. It is crucial to critically evaluate the information provided in these responses,"
    " recognizing that some of it may be biased or incorrect. Your response should not simply"
    " replicate the given answers but should offer a refined, accurate, and comprehensive"
    " reply to the instruction. Ensure your response is well-structured, coherent, and adheres"
    " to the highest standards of accuracy and reliability."
    "\nResponses from models:"
)


class MixtureOfAgents(AsyncLLM):
    aggregator_llm: LLM
    proposers_llms: List[AsyncLLM]
    rounds: int = 1

    @field_validator("proposers_llms")
    @classmethod
    def ensure_at_least_one_proposer_llm(
        cls, proposers_llms: List[AsyncLLM]
    ) -> List[AsyncLLM]:
        """Ensures that the `MixtureOfAgents` has at least one `LLM` in `proposers_llms`.

        Args:
            proposers_llms: The list of `LLM`s.

        Returns:
            The list of `LLM`s.
        """
        if len(proposers_llms) == 0:
            raise ValueError(
                "`MixtureOfAgents` must have at least one `LLM` in `propose_llms`."
            )

        return proposers_llms

    @property
    def runtime_parameters_names(self) -> "RuntimeParametersNames":
        """Returns the runtime parameters of the `LLM`, which are a combination of the
        `RuntimeParameter`s of the `LLM`, the `aggregator_llm` and the `proposers_llms`.

        Returns:
            The runtime parameters of the `LLM`.
        """
        runtime_parameters_names = super().runtime_parameters_names
        del runtime_parameters_names["generation_kwargs"]
        return runtime_parameters_names

    def load(self) -> None:
        """Loads all the `LLM`s in the `MixtureOfAgents`."""
        super().load()

        for llm in self.proposers_llms:
            self._logger.debug(f"Loading proposer LLM in MoA: {llm}")  # type: ignore
            llm.load()

        self._logger.debug(f"Loading aggregator LLM in MoA: {self.aggregator_llm}")  # type: ignore
        self.aggregator_llm.load()

    @property
    def model_name(self) -> str:
        """Returns the aggregated model name."""
        return f"moa-{self.aggregator_llm.model_name}-{'-'.join([llm.model_name for llm in self.proposers_llms])}"

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Returns the generation kwargs of the `MixtureOfAgents` as a dictionary.

        Returns:
            The generation kwargs of the `MixtureOfAgents`.
        """
        return {
            "aggregator_llm": self.aggregator_llm.get_generation_kwargs(),
            "proposers_llms": [
                llm.get_generation_kwargs() for llm in self.proposers_llms
            ],
        }

    # `abstractmethod`, had to be implemented but not used
    async def agenerate(
        self, input: "FormattedInput", num_generations: int = 1, **kwargs: Any
    ) -> List[Union[str, None]]:
        raise NotImplementedError(
            "`agenerate` method is not implemented for `MixtureOfAgents`"
        )

    def _build_moa_system_prompt(self, prev_outputs: List[str]) -> str:
        moa_system_prompt = MOA_SYSTEM_PROMPT
        for i, prev_output in enumerate(prev_outputs):
            if prev_output is not None:
                moa_system_prompt += f"\n{i + 1}. {prev_output}"
        return moa_system_prompt

    def _inject_moa_system_prompt(
        self, input: "StandardInput", prev_outputs: List[str]
    ) -> "StandardInput":
        if len(prev_outputs) == 0:
            return input

        moa_system_prompt = self._build_moa_system_prompt(prev_outputs)

        system = next((item for item in input if item["role"] == "system"), None)
        if system:
            system["content"] = moa_system_prompt
        else:
            input.insert(0, {"role": "system", "content": moa_system_prompt})

        return input

    async def _agenerate(
        self,
        inputs: List["FormattedInput"],
        num_generations: int = 1,
        **kwargs: Any,
    ) -> List["GenerateOutput"]:
        aggregator_llm_kwargs: Dict[str, Any] = kwargs.get("aggregator_llm", {})
        proposers_llms_kwargs: List[Dict[str, Any]] = kwargs.get("proposers_llms", [])

        prev_outputs = []
        for round in range(self.rounds):
            self._logger.debug(f"Generating round {round + 1}/{self.rounds} in MoA")  # type: ignore

            # Generate `num_generations` with each proposer LLM for each input
            tasks = [
                asyncio.create_task(
                    llm._agenerate(
                        inputs=[
                            self._inject_moa_system_prompt(
                                cast("StandardInput", input), prev_input_outputs
                            )
                            for input, prev_input_outputs in itertools.zip_longest(
                                inputs, prev_outputs, fillvalue=[]
                            )
                        ],
                        num_generations=1,
                        **generation_kwargs,
                    )
                )
                for llm, generation_kwargs in zip(
                    self.proposers_llms, proposers_llms_kwargs
                )
            ]

            # Group generations per input
            outputs: List[List["GenerateOutput"]] = await asyncio.gather(*tasks)
            prev_outputs = [
                list(itertools.chain(*input_outputs)) for input_outputs in zip(*outputs)
            ]

        self._logger.debug("Aggregating outputs in MoA")  # type: ignore
        if isinstance(self.aggregator_llm, AsyncLLM):
            return await self.aggregator_llm._agenerate(
                inputs=[
                    self._inject_moa_system_prompt(
                        cast("StandardInput", input), prev_input_outputs
                    )
                    for input, prev_input_outputs in zip(inputs, prev_outputs)
                ],
                num_generations=num_generations,
                **aggregator_llm_kwargs,
            )

        return self.aggregator_llm.generate(
            inputs=[
                self._inject_moa_system_prompt(
                    cast("StandardInput", input), prev_input_outputs
                )
                for input, prev_input_outputs in zip(inputs, prev_outputs)
            ],
            num_generations=num_generations,
            **aggregator_llm_kwargs,
        )
