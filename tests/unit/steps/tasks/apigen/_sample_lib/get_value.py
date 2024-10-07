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

from typing import List, Optional, Tuple


def get_value(matrix: List[List[int]], indices: Tuple[int, int]) -> Optional[int]:
    """Gets the value at the specified index in the matrix.

    Args:
        matrix: A list of lists representing the matrix.
        indices: A tuple containing the row and column indices.
    """
    row_index, col_index = indices
    if (
        row_index < 0
        or row_index >= len(matrix)
        or col_index < 0
        or col_index >= len(matrix[row_index])
    ):
        return None
    return matrix[row_index][col_index]
