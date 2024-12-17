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

import re
from typing import TYPE_CHECKING, Any, Dict, Literal, Union

from distilabel.steps.tasks.base import Task

if TYPE_CHECKING:
    from distilabel.steps.tasks.typing import FormattedInput
    from distilabel.steps.typing import StepColumns

GSM8K_AUGMENTATION_SYSTEM_PROMPT = """
I want you to act as a math teacher. I will provide a grade school math question and you will help to create a more challenging math questions by given ways. Given the question: "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", you will modify it by following ideas:

1. **Change specific numbers**: James writes a 2-page letter to 2 different friends 3 times a week. How many pages does he write in 4 years?
2. **Introduce fractions or percentages**: James writes a 3-page letter to 2 different friends twice a week. Each week, he adds 50% more pages to each letter. How many pages does he write in a month?
3. **Combine multiple concepts**: James writes a 3-page letter to 2 different friends twice a week. He uses both sides of the paper and each side can hold 250 words. If James writes 100 words per minute, how long does it take for him to write all the letters in a week?
4. **Include a conditional statement**: James writes a 3-page letter to 2 different friends twice a week. If it's holiday, he writes an additional 5-page letter to each friend. Considering there are 10 holidays in a year, how many pages does he write in a year?
5. **Increase the complexity of the problem**: James writes a 3-page letter to 2 different friends twice a week. In addition, he writes a 5-page letter to 3 other friends once a week. How many pages does he write in a month, assuming there are 4 weeks in a month?

Now you are given the question:
""".lstrip()

EXTRACT_REGEX = re.compile(r"^\d+\.\s+\*\*[^*]+\*\*:\s*(.*?)$", re.MULTILINE)

MATH_AUGMENTATION_SYSTEM_PROMPT = r"""
I want you to act as a math teacher. You should think of some ways to help students do variation training for challenging competition mathematics problems. For example, for a question-solution pair:

Question0: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write in a year?

Solution0: He writes each friend $3 \times 2 = 6$ pages a week. So he writes $6 \times 2 = 12$ pages every week. That means he writes $12 \times 52 = \boxed{624}$ pages a year ##end0

We can propose 5 types of variation exercises, and response with:

1. Change specific numbers:
Question1: James writes a 5-page letter to 3 different friends 4 times a week. How many pages does he write in 3 years?
Solution1: To calculate the total number of pages James writes in 3 years, let's first figure out how many pages he writes each week and then multiply that by the number of weeks in 3 years. He writes each friend a 5-page letter, so for 3 friends, that's $5 \times 3 = 15$ pages per writing session. He writes 4 times a week, so the weekly total is $15 \times 4 = 60$ pages. There are 52 weeks in a year, so in one year, he writes $60 \times 52 = 3120$ pages. Finally, over the course of 3 years, he writes $3120 \times 3 = \boxed{9360}$ pages. ##end1

2. Introduce fractions or percentages:
Question2: James writes a 3-page letter to 2 different friends twice a week. Each week, he adds 100% more pages to each letter. How many pages does he write in a month?
Solution2: Let's take this step by step: In the first week, James writes a 3-page letter to 2 friends twice a week, which is $3 \times 2 \times 2 = 12$ pages in total for the first week. In the second week, he writes 100% more pages, thus doubling the number of pages in each letter. So he writes $6 \times 2 \times 2 = 24$ pages in total for the second week. In the third week, he again writes double the previous week's pages, so $12 \times 2 \times 2 = 48$ pages in total for the third week. In the fourth week, the number of pages doubles again, which results in $24 \times 2 \times 2 = 96$ pages in total for the fourth week. Now, we'll add up the pages from all four weeks to find out how many pages he writes in a month: $12 + 24 + 48 + 96 = \boxed{180}$ pages. Therefore, in a month (assuming a 4-week month), James writes 180 pages. ##end2

3. Combine multiple concepts:
Question3: James writes a 3-page letter to 2 different friends twice a week. He uses both sides of the paper, and each side can hold 250 words. If James writes at a speed of 100 words per minute, how long does it take him to write all the letters in a week?
Solution3: To find out how long it takes James to write all the letters in a week, we first calculate how many words he writes in total. Each letter is 3 pages long, and he writes to 2 friends, which is $3 \times 2 = 6$ pages per writing session. Since he writes twice a week, the total number of pages per week is $6 \times 2 = 12$ pages. Considering each page has two sides and each side holds 250 words, the number of words on one page is $250 \times 2 = 500$ words. Therefore, the total number of words James writes in a week is $500 \times 12 = 6000$ words. Given James writes at a speed of 100 words per minute, the time it takes him to write all the letters in a week is calculated by dividing the total number of words by his writing speed: $6000 \div 100 = 60$ minutes. So, James takes 60 minutes to write all the letters in a week. ##end3

4. Include a conditional statement:
Question4: XX
Solution4: XX ##end4

5. Increase the complexity of the problem:
Question5: XX
Solution5: XX ##end5

Now, find five suitable variation training methods for the new problem. Be careful not to let existing methods limit your thinking. Instead, propose variation training methods that are specifically tailored to the given problem:
""".lstrip()

MATH_AUGMENTATION_USER_MESSAGE = "Question0: {question}\nSolution0: {solution}\nPlease response with the given example format(including Questions and solutions)"


class MuggleMathAugmentQuery(Task):
    query_augmentation_prompt: Literal["gsm8k", "math"]

    @property
    def inputs(self) -> "StepColumns":
        if self.query_augmentation_prompt == "gsm8k":
            return ["question"]

        return ["question", "solution"]

    def format_input(self, input: Dict[str, Any]) -> "FormattedInput":
        system = (
            GSM8K_AUGMENTATION_SYSTEM_PROMPT
            if self.query_augmentation_prompt == "gsm8k"
            else MATH_AUGMENTATION_SYSTEM_PROMPT
        )

        question = input["question"]
        solution = input.get("solution")

        user = (
            question
            if self.query_augmentation_prompt == "gsm8k"
            else MATH_AUGMENTATION_USER_MESSAGE.format(
                question=question, solution=solution
            )
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @property
    def outputs(self) -> "StepColumns":
        if self.query_augmentation_prompt == "gsm8k":
            return ["augmented_questions"]

        return ["augmented_questions", "solutions"]

    def format_output(
        self, output: Union[str, None], input: Union[Dict[str, Any], None] = None
    ) -> Dict[str, Any]:
        if output is None:
            return {"augmented_questions": None}

        matches = EXTRACT_REGEX.finditer(output)
        questions = [match.group(1) for match in matches]

        if self.query_augmentation_prompt == "gsm8k":
            return {"augmented_questions": questions}

        return {"augmented_questions": questions, "solutions": []}
