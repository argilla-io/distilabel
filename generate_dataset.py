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


from datasets import load_dataset

dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")

# Remove the columns that are not needed
dataset = dataset.remove_columns(["input"])

# Get the first 10 rows
dataset = dataset["train"].select(range(10))

# Push this dataframe to HF datasets
dataset.push_to_hub("ignacioct/instruction_examples", "main")
