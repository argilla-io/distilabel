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

from distilabel.errors import DistilabelUserError


def check_column_in_template(
    column: str, template: str, page: str = "components-gallery/tasks/textgeneration/"
) -> None:
    """Checks if a column is present in the template, and raises an error if it isn't.

    Args:
        column: The column name to check in the template.
        template: The template of the Task to be checked, the input from the user.
        page: The page to redirect the user for help . Defaults to "components-gallery/tasks/textgeneration/".

    Raises:
        DistilabelUserError: Custom error if the column is not present in the template.
    """
    pattern = (
        r"(?:{%.*?\b"
        + re.escape(column)
        + r"\b.*?%}|{{\s*"
        + re.escape(column)
        + r"\s*}})"
    )
    if not re.search(pattern, template):
        raise DistilabelUserError(
            (
                f"You required column name '{column}', but is not present in the template, "
                "ensure the 'columns' match with the 'template' to avoid errors."
            ),
            page=page,
        )
