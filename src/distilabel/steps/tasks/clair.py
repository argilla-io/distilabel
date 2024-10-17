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

import importlib.resources as importlib_resources
from typing import TYPE_CHECKING, Any, Dict, Final, Union

from jinja2 import Template
from pydantic import PrivateAttr

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import ChatType
    from distilabel.steps.typing import StepColumns


SYSTEM_PROMPT: Final[str] = (
    "You are a teacher and your task is to minimally improve a student's answer. I will give you a {task} and a {student_solution}. Your job is to revise the {student_solution} such that it is clearer, more correct, and more engaging. Copy all non-corrected parts of the student's answer. Do not allude to the {corrected_student_solution} being a revision or a correction in your final solution."
)


class CLAIR(Task):
    r"""Contrastive Learning from AI Revisions (CLAIR).

    CLAIR uses an AI system to minimally revise a solution A→A´ such that the resulting
    preference A `preferred` A’ is much more contrastive and precise.

    Input columns:
        - task (`str`): The task or instruction.
        - student_solution (`str`): An answer to the task that is to be revised.

    Output columns:
        - revision (`str`): The revised text.
        - rational (`str`): The rational for the provided revision.
        - model_name (`str`): The name of the model used to generate the revision and rational.

    Categories:
        - preference
        - text-generation

    References:
        - [`Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment`](https://arxiv.org/abs/2408.06266v1)
        - [`APO and CLAIR - GitHub Repository`](https://github.com/ContextualAI/CLAIR_and_APO)

    Examples:
        Create contrastive preference pairs:

        ```python
        from distilabel.steps.tasks import CLAIR
        from distilabel.llms.huggingface import InferenceEndpointsLLM

        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 4096,
            },
        )
        clair_task = CLAIR(llm=llm)

        clair_task.load()

        result = next(
            clair_task.process(
                [
                    {
                        "task": "How many gaps are there between the earth and the moon?",
                        "student_solution": 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon's orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.'
                    }
                ]
            )
        )
        # result
        # [{'task': 'How many gaps are there between the earth and the moon?',
        # 'student_solution': 'There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.',
        # 'revision': 'There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
        # 'rational': 'The student\'s solution provides a clear and concise answer to the question. However, there are a few areas where it can be improved. Firstly, the term "gaps" can be misleading in this context. The student should clarify what they mean by "gaps." Secondly, the student provides some additional information about the Moon\'s orbit, which is correct but could be more clearly connected to the main point. Lastly, the student\'s conclusion could be more concise.',
        # 'distilabel_metadata': {'raw_output_c_l_a_i_r_0': '{teacher_reasoning}: The student\'s solution provides a clear and concise answer to the question. However, there are a few areas where it can be improved. Firstly, the term "gaps" can be misleading in this context. The student should clarify what they mean by "gaps." Secondly, the student provides some additional information about the Moon\'s orbit, which is correct but could be more clearly connected to the main point. Lastly, the student\'s conclusion could be more concise.\n\n{corrected_student_solution}: There are no physical gaps or empty spaces between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a significant separation or gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range. This variation in distance is a result of the Moon\'s orbital path, not the presence of any gaps.\n\nIn summary, the Moon\'s orbit is continuous, with no intervening gaps, and its distance from the Earth varies due to the elliptical shape of its orbit.',
        # 'raw_input_c_l_a_i_r_0': [{'role': 'system',
        #     'content': "You are a teacher and your task is to minimally improve a student's answer. I will give you a {task} and a {student_solution}. Your job is to revise the {student_solution} such that it is clearer, more correct, and more engaging. Copy all non-corrected parts of the student's answer. Do not allude to the {corrected_student_solution} being a revision or a correction in your final solution."},
        #     {'role': 'user',
        #     'content': '{task}: How many gaps are there between the earth and the moon?\n\n{student_solution}: There are no gaps between the Earth and the Moon. The Moon is actually in a close orbit around the Earth, and it is held in place by gravity. The average distance between the Earth and the Moon is about 384,400 kilometers (238,900 miles), and this distance is known as the "lunar distance" or "lunar mean distance."\n\nThe Moon does not have a gap between it and the Earth because it is a natural satellite that is gravitationally bound to our planet. The Moon\'s orbit is elliptical, which means that its distance from the Earth varies slightly over the course of a month, but it always remains within a certain range.\n\nSo, to summarize, there are no gaps between the Earth and the Moon. The Moon is simply a satellite that orbits the Earth, and its distance from our planet varies slightly due to the elliptical shape of its orbit.\n\n-----------------\n\nLet\'s first think step by step with a {teacher_reasoning} to decide how to improve the {student_solution}, then give the {corrected_student_solution}. Mention the {teacher_reasoning} and {corrected_student_solution} identifiers to structure your answer.'}]},
        # 'model_name': 'meta-llama/Meta-Llama-3.1-70B-Instruct'}]
        ```

    Citations:

        ```
        @misc{doosterlinck2024anchoredpreferenceoptimizationcontrastive,
            title={Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment},
            author={Karel D'Oosterlinck and Winnie Xu and Chris Develder and Thomas Demeester and Amanpreet Singh and Christopher Potts and Douwe Kiela and Shikib Mehri},
            year={2024},
            eprint={2408.06266},
            archivePrefix={arXiv},
            primaryClass={cs.LG},
            url={https://arxiv.org/abs/2408.06266},
        }
        ```
    """

    system_prompt: str = SYSTEM_PROMPT
    _template: Union[Template, None] = PrivateAttr(...)

    def load(self) -> None:
        super().load()
        _path = str(
            importlib_resources.files("distilabel")
            / "steps"
            / "tasks"
            / "templates"
            / "clair.jinja2"
        )
        with open(_path, "r") as f:
            self._template = Template(f.read())

    @property
    def inputs(self) -> "StepColumns":
        return ["task", "student_solution"]

    @property
    def outputs(self) -> "StepColumns":
        return ["revision", "rational", "model_name"]

    def format_input(self, input: Dict[str, Any]) -> "ChatType":
        """The input is formatted as a `ChatType` assuming that the instruction
        is the first interaction from the user within a conversation."""
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self._template.render(
                    task=input["task"], student_solution=input["student_solution"]
                ),
            },
        ]

    def format_output(
        self, output: Union[str, None], input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """The output is formatted as a list with the score of each instruction-response pair.

        Args:
            output: the raw output of the LLM.
            input: the input to the task. Used for obtaining the number of responses.

        Returns:
            A dict with the key `scores` containing the scores for each instruction-response pair.
        """
        if output is None:
            return self._default_error()

        return self._format_output(output)

    def _format_output(self, output: Union[str, None]) -> Dict[str, Any]:
        if "**Corrected Student Solution:**" in output:
            splits = output.split("**Corrected Student Solution:**")
        elif "{corrected_student_solution}:" in output:
            splits = output.split("{corrected_student_solution}:")
        elif "{corrected_student_solution}" in output:
            splits = output.split("{corrected_student_solution}")
        elif "**Worsened Student Solution:**" in output:
            splits = output.split("**Worsened Student Solution:**")
        elif "{worsened_student_solution}:" in output:
            splits = output.split("{worsened_student_solution}:")
        elif "{worsened_student_solution}" in output:
            splits = output.split("{worsened_student_solution}")
        else:
            splits = None

        # Safety check when the output doesn't follow the expected format
        if not splits:
            return self._default_error()

        if len(splits) >= 2:
            revision = splits[1]
            revision = revision.strip("\n\n").strip()  # noqa: B005

            rational = splits[0]
            if "{teacher_reasoning}" in rational:
                rational = rational.split("{teacher_reasoning}")[1].strip(":").strip()
            rational = rational.strip("\n\n").strip()  # noqa: B005
        else:
            return self._default_error()
        return {"revision": revision, "rational": rational}

    def _default_error(self) -> Dict[str, None]:
        return {"revision": None, "rational": None}
