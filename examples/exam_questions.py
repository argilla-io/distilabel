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

from typing import List

import wikipedia
from pydantic import BaseModel, Field

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

page = wikipedia.page(title="Transfer_learning")


class ExamQuestion(BaseModel):
    question: str = Field(..., description="The question to be answered")
    answer: str = Field(..., description="The correct answer to the question")
    distractors: List[str] = Field(
        ..., description="A list of incorrect but viable answers to the question"
    )


class ExamQuestions(BaseModel):
    exam: List[ExamQuestion]


SYSTEM_PROMPT = """\
You are an exam writer specialized in writing exams for students.
Your goal is to create questions and answers based on the document provided, and a list of distractors, that are incorrect but viable answers to the question.
Your answer must adhere to the following format:
```
[
    {
        "question": "Your question",
        "answer": "The correct answer to the question",
        "distractors": ["wrong answer 1", "wrong answer 2", "wrong answer 3"]
    },
    ... (more questions and answers as required)
]
```
""".strip()


with Pipeline(name="ExamGenerator") as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_instructions",
        data=[
            {
                "page": page.content,
            }
        ],
    )

    text_generation = TextGeneration(
        name="exam_generation",
        system_prompt=SYSTEM_PROMPT,
        template="Generate a list of answers and questions about the document. Document:\n\n{{ page }}",
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            structured_output={
                "schema": ExamQuestions.model_json_schema(),
                "format": "json",
            },
        ),
        input_batch_size=8,
        output_mappings={"model_name": "generation_model"},
    )
    load_dataset >> text_generation


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 2048,
                    }
                }
            }
        },
        use_cache=False,
    )
    distiset.push_to_hub("USERNAME/exam_questions")
