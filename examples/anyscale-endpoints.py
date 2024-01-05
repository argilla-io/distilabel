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

from datasets import load_dataset
from distilabel.llm import AnyscaleLLM
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask

if __name__ == "__main__":
    dataset = (
        load_dataset("HuggingFaceH4/instruction-dataset", split="test[10:12]")
        .remove_columns(["completion", "meta"])
        .rename_column("prompt", "input")
    )
    pipe = Pipeline(
        generator=AnyscaleLLM(
            model="HuggingFaceH4/zephyr-7b-beta",
            task=TextGenerationTask(),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    )
    new_dataset = pipe.generate(
        dataset,  # type: ignore
        num_generations=1,
    )

    print(new_dataset[0])
    # {'generation_model': ['HuggingFaceH4/zephyr-7b-beta'],
    # 'generation_prompt': [[{'content': 'You are a helpful, respectful and honest '
    #                                     'assistant. Always answer as helpfully as '
    #                                     'possible, while being safe. Your answers '
    #                                     'should not include any harmful, '
    #                                     'unethical, racist, sexist, toxic, '
    #                                     'dangerous, or illegal content. Please '
    #                                     'ensure that your responses are socially '
    #                                     'unbiased and positive in nature.\n'
    #                                     'If a question does not make any sense, or '
    #                                     'is not factually coherent, explain why '
    #                                     'instead of answering something not '
    #                                     "correct. If you don't know the answer to "
    #                                     "a question, please don't share false "
    #                                     'information.',
    #                         'role': 'system'},
    #                         {'content': 'Does the United States use Celsius or '
    #                                     'Fahrenheit?',
    #                         'role': 'user'}]],
    # 'generations': ['The United States primarily uses Fahrenheit for measuring '
    #                 'temperature in most everyday applications. However, some '
    #                 'scientific fields, such as meteorology and scientific '
    #                 'research, use Celsius as well, particularly when working '
    #                 'with international collaborations or sharing data globally. '
    #                 'Other countries that use Fahrenheit are Belize, the Cayman '
    #                 'Islands, and Liberia. Overall, Celsius is more commonly used '
    #                 'internationally.'],
    # 'input': 'Does the United States use Celsius or Fahrenheit?',
    # 'raw_generation_responses': ['The United States primarily uses Fahrenheit for '
    #                             'measuring temperature in most everyday '
    #                             'applications. However, some scientific fields, '
    #                             'such as meteorology and scientific research, '
    #                             'use Celsius as well, particularly when working '
    #                             'with international collaborations or sharing '
    #                             'data globally. Other countries that use '
    #                             'Fahrenheit are Belize, the Cayman Islands, and '
    #                             'Liberia. Overall, Celsius is more commonly used '
    #                             'internationally.']}
