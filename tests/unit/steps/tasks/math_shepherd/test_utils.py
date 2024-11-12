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

from distilabel.steps import ExpandColumns
from distilabel.steps.tasks.math_shepherd.utils import FormatPRM


class TestFormatPRM:
    def test_process(self) -> None:
        # As the expected data format is a JSON encoded string of a list, will use
        # ExpandColumns initially as that would be a prior step in the pipeline
        expand_columns = ExpandColumns(
            columns=["solutions"],
            encoded=True,
        )
        expand_columns.load()
        result = next(
            expand_columns.process(
                [
                    {
                        "instruction": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                        "solutions": '[["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can multiply 2 by 0.5 (which is the same as dividing by 2): 2 * 0.5 = <<2*0.5=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"], ["Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +", "Step 2: Calculate the amount of white fiber needed: Since it\'s half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +", "Step 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"]]',
                    }
                ]
            )
        )

        formatter = FormatPRM()
        formatter.load()
        result = next(formatter.process(result))
        assert (
            result[0]["input"]
            == "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. ки\nStep 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. ки\nStep 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 ки"
        )
        assert (
            result[0]["label"]
            == "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Step 1: Determine the amount of blue fiber needed: 2 bolts of blue fiber are required. +\nStep 2: Calculate the amount of white fiber needed: Since it's half that much, we can divide 2 by 2: 2 / 2 = <<2/2=1>>1 bolt of white fiber. +\nStep 3: Add the amount of blue and white fiber: 2 (blue) + 1 (white) = <<2+1=3>>3 bolts of fiber in total. The answer is: 3 +"
        )
