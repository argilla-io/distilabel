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

import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

from typing import Any, Dict, Union

import pytest
from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.prometheus_eval import (
    _RUBRICS,
    PrometheusAbsEval,
    PrometheusRelEval,
)
from jinja2 import Template

from tests.unit.steps.tasks.utils import DummyLLM


def load_template(template: str) -> Template:
    return Template(
        open(
            str(
                importlib_resources.files("distilabel")
                / "steps/tasks/templates/prometheus"
                / template
            )
        ).read()
    )


class TestPrometheusAbsEval:
    @pytest.mark.parametrize(
        "rubric, reference, template, input",
        [
            (
                "helpfulness",
                True,
                "absolute_with_reference.jinja2",
                {"instruction": "A", "generation": "B", "reference": "C"},
            ),
            (
                "harmlessness",
                False,
                "absolute_without_reference.jinja2",
                {"instruction": "A", "generation": "B"},
            ),
            (
                "honesty",
                True,
                "absolute_with_reference.jinja2",
                {"instruction": "A", "generation": "B", "reference": "C"},
            ),
            (
                "factual-validity",
                False,
                "absolute_without_reference.jinja2",
                {"instruction": "A", "generation": "B"},
            ),
            (
                "reasoning",
                True,
                "absolute_with_reference.jinja2",
                {"instruction": "A", "generation": "B", "reference": "C"},
            ),
        ],
    )
    def test_format_input(
        self, rubric: str, reference: bool, template: str, input: Dict[str, str]
    ) -> None:
        task = PrometheusAbsEval(
            name="task",
            rubric=rubric,  # type: ignore
            reference=reference,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        template_kwargs = input
        template_kwargs["rubric"] = _RUBRICS[rubric]

        assert task.format_input(input=input)[-1]["content"] == load_template(
            template=template
        ).render(**template_kwargs)

    def test_format_input_errors(self) -> None:
        # `reference=True` but reference not provided
        task = PrometheusAbsEval(
            name="task",
            rubric="helpfulness",
            reference=True,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        with pytest.raises(KeyError, match="reference"):
            task.format_input(input={"instruction": "A", "generation": "B"})

    @pytest.mark.parametrize(
        "output, expected",
        [
            (
                "Feedback: A \n[RESULT] 1\n",
                {"feedback": "A", "result": 1},
            ),
            (
                "Feedback: A [RESULT] 1",
                {"feedback": "A", "result": 1},
            ),
            (
                "Feedback: A [RESULT] 6",
                {"feedback": None, "result": None},
            ),
            (
                "A [RESULT] 1",
                {"feedback": "A", "result": 1},
            ),
            (
                None,
                {"feedback": None, "result": None},
            ),
        ],
    )
    def test_format_output(
        self, output: Union[str, None], expected: Dict[str, Any]
    ) -> None:
        task = PrometheusAbsEval(
            name="task",
            rubric="factual-validity",
            reference=False,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert (
            task.format_output(
                output=output,
                input={
                    "instruction": "A",
                    "generation": "B",
                },
            )
            == expected
        )


class TestPrometheusRelEval:
    @pytest.mark.parametrize(
        "rubric, reference, template, input",
        [
            (
                "helpfulness",
                True,
                "relative_with_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"], "reference": "D"},
            ),
            (
                "harmlessness",
                False,
                "relative_without_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"]},
            ),
            (
                "honesty",
                True,
                "relative_with_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"], "reference": "D"},
            ),
            (
                "factual-validity",
                False,
                "relative_without_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"]},
            ),
            (
                "reasoning",
                True,
                "relative_with_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"], "reference": "D"},
            ),
        ],
    )
    def test_format_input(
        self, rubric: str, reference: bool, template: str, input: Dict[str, str]
    ) -> None:
        task = PrometheusRelEval(
            name="task",
            rubric=rubric,  # type: ignore
            reference=reference,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        template_kwargs = input
        template_kwargs["rubric"] = _RUBRICS[rubric]

        assert task.format_input(input=input)[-1]["content"] == load_template(
            template=template
        ).render(**template_kwargs)

    def test_format_input_errors(self) -> None:
        # `reference=True` but reference not provided
        task = PrometheusRelEval(
            name="task",
            rubric="helpfulness",
            reference=True,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        with pytest.raises(KeyError, match="reference"):
            task.format_input(input={"instruction": "A", "generations": ["B", "C"]})

        # `generations` is not a list with exactly 2 elements
        task = PrometheusRelEval(
            name="task",
            rubric="helpfulness",
            reference=False,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        for generations in [[], ["A"], ["A", "B", "C"]]:
            with pytest.raises(
                ValueError,
                match=r"Provided \`generations\` is of type \<class 'list'\> but a list of strings with length 2 should be provided instead",
            ):
                task.format_input(
                    input={"instruction": "A", "generations": generations}
                )

    @pytest.mark.parametrize(
        "output, expected",
        [
            (
                "Feedback: A \n[RESULT] 1\n",
                {"feedback": None, "result": None},
            ),
            (
                "Feedback: A [RESULT] A",
                {"feedback": "A", "result": "A"},
            ),
            (
                "Feedback: A [RESULT] B",
                {"feedback": "A", "result": "B"},
            ),
            (
                "A [RESULT] 1",
                {"feedback": None, "result": None},
            ),
            (
                None,
                {"feedback": None, "result": None},
            ),
        ],
    )
    def test_format_output(
        self, output: Union[str, None], expected: Dict[str, Any]
    ) -> None:
        task = PrometheusRelEval(
            name="task",
            rubric="factual-validity",
            reference=False,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert (
            task.format_output(
                output=output,
                input={
                    "instruction": "A",
                    "generations": ["B", "C"],
                },
            )
            == expected
        )
