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
from jinja2 import Template
from pydantic import ValidationError

from distilabel.pipeline.local import Pipeline
from distilabel.steps.tasks.prometheus_eval import _DEFAULT_RUBRICS, PrometheusEval
from tests.unit.conftest import DummyLLM


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
        "mode, rubric, reference, template, input",
        [
            (
                "absolute",
                "helpfulness",
                True,
                "absolute_with_reference.jinja2",
                {"instruction": "A", "generation": "B", "reference": "C"},
            ),
            (
                "absolute",
                "harmlessness",
                False,
                "absolute_without_reference.jinja2",
                {"instruction": "A", "generation": "B"},
            ),
            (
                "absolute",
                "honesty",
                True,
                "absolute_with_reference.jinja2",
                {"instruction": "A", "generation": "B", "reference": "C"},
            ),
            (
                "absolute",
                "factual-validity",
                False,
                "absolute_without_reference.jinja2",
                {"instruction": "A", "generation": "B"},
            ),
            (
                "absolute",
                "reasoning",
                True,
                "absolute_with_reference.jinja2",
                {"instruction": "A", "generation": "B", "reference": "C"},
            ),
            (
                "relative",
                "helpfulness",
                True,
                "relative_with_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"], "reference": "D"},
            ),
            (
                "relative",
                "harmlessness",
                False,
                "relative_without_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"]},
            ),
            (
                "relative",
                "honesty",
                True,
                "relative_with_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"], "reference": "D"},
            ),
            (
                "relative",
                "factual-validity",
                False,
                "relative_without_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"]},
            ),
            (
                "relative",
                "reasoning",
                True,
                "relative_with_reference.jinja2",
                {"instruction": "A", "generations": ["B", "C"], "reference": "D"},
            ),
        ],
    )
    def test_format_input(
        self,
        mode: str,
        rubric: str,
        reference: bool,
        template: str,
        input: Dict[str, str],
    ) -> None:
        task = PrometheusEval(
            name="task",
            mode=mode,  # type: ignore
            rubric=rubric,  # type: ignore
            reference=reference,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        template_kwargs = input
        template_kwargs["rubric"] = _DEFAULT_RUBRICS[rubric]

        assert task.format_input(input=input)[-1]["content"] == load_template(
            template=template
        ).render(**template_kwargs)

    def test_format_input_errors(self) -> None:
        # any `mode` and `reference=True` but reference not provided
        task = PrometheusEval(
            name="task",
            mode="absolute",
            rubric="helpfulness",
            reference=True,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        with pytest.raises(KeyError, match="reference"):
            task.format_input(input={"instruction": "A", "generation": "B"})

        # `mode=absolute` and `generation` is not a string
        task = PrometheusEval(
            name="task",
            mode="absolute",
            rubric="helpfulness",
            reference=False,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        with pytest.raises(
            ValueError,
            match=r"Provided \`generation\` is of type \<class 'int'\> but a string should be provided instead.",
        ):
            task.format_input(input={"instruction": "A", "generation": 1})

        # `mode=relative` and `generations` is not a list with exactly 2 elements
        task = PrometheusEval(
            name="task",
            mode="relative",
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
        "mode, output, expected",
        [
            (
                "absolute",
                "Feedback: A \n[RESULT] 1\n",
                {"feedback": "A", "result": 1},
            ),
            (
                "absolute",
                "Feedback: A [RESULT] 1",
                {"feedback": "A", "result": 1},
            ),
            (
                "absolute",
                "Feedback: A [RESULT] 6",
                {"feedback": None, "result": None},
            ),
            (
                "absolute",
                "A [RESULT] 1",
                {"feedback": "A", "result": 1},
            ),
            (
                "absolute",
                None,
                {"feedback": None, "result": None},
            ),
            (
                "relative",
                "Feedback: A \n[RESULT] 1\n",
                {"feedback": None, "result": None},
            ),
            (
                "relative",
                "Feedback: A [RESULT] A",
                {"feedback": "A", "result": "A"},
            ),
            (
                "relative",
                "Feedback: A [RESULT] B",
                {"feedback": "A", "result": "B"},
            ),
            (
                "relative",
                "A [RESULT] 1",
                {"feedback": None, "result": None},
            ),
            (
                "relative",
                None,
                {"feedback": None, "result": None},
            ),
        ],
    )
    def test_format_output(
        self, mode: str, output: Union[str, None], expected: Dict[str, Any]
    ) -> None:
        task = PrometheusEval(
            name="task",
            mode=mode,  # type: ignore
            rubric="factual-validity",
            reference=False,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )
        task.load()

        assert task.format_output(output=output, input={}) == expected

    def test_custom_rubrics(self) -> None:
        # As we're using a `pydantic.BaseModel` underneath, we are using a custom
        # `model_validator` after the attributes are set, so if either the `rubric`
        # or the provided `rubrics` are wrong, the `model_validator` will raise an
        # error.
        PrometheusEval(
            name="task",
            mode="absolute",
            rubric="custom",
            rubrics={
                "custom": "[A]\nScore 1: A\nScore 2: B\nScore 3: C\nScore 4: D\nScore 5: E"
            },
            reference=False,
            llm=DummyLLM(),
            pipeline=Pipeline(name="unit-test-pipeline"),
        )

    def test_custom_rubrics_errors(self) -> None:
        # 1. `rubrics` is not a valid dict
        with pytest.raises(
            ValidationError,
            match=r"Provided \`rubrics\` must be a Python dictionary with string keys and string values.",
        ):
            PrometheusEval(
                name="task",
                mode="absolute",
                rubric="custom",
                rubrics={},
                reference=False,
                llm=DummyLLM(),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
        with pytest.raises(
            ValidationError,
            match=r"rubrics.custom\n  Input should be a valid string",
        ):
            PrometheusEval(
                name="task",
                mode="absolute",
                rubric="custom",
                rubrics={"custom": 1},
                reference=False,
                llm=DummyLLM(),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
        # 2. `rubrics` is not compliant with the pre-defined schema
        with pytest.raises(
            ValidationError,
            match=r"Provided rubrics should match the format of the default rubrics,",
        ):
            PrometheusEval(
                name="task",
                mode="absolute",
                rubric="custom",
                rubrics={"custom": "wrong schema"},
                reference=False,
                llm=DummyLLM(),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
        # 3. `rubric` is not available in `rubrics`
        with pytest.raises(
            ValidationError,
            match=r"Provided rubric 'wrong' is not among the available rubrics: custom.",
        ):
            PrometheusEval(
                name="task",
                mode="absolute",
                rubric="wrong",
                rubrics={
                    "custom": "[A]\nScore 1: A\nScore 2: B\nScore 3: C\nScore 4: D\nScore 5: E"
                },
                reference=False,
                llm=DummyLLM(),
                pipeline=Pipeline(name="unit-test-pipeline"),
            )
