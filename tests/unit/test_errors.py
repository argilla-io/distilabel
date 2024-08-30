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

from distilabel.errors import DistilabelUserError


def test_distilabel_user_error() -> None:
    msg = DistilabelUserError("This is an error message.")
    assert str(msg) == "This is an error message."
    msg = DistilabelUserError(
        "This is an error message.", page="sections/getting_started/faq/"
    )
    assert (
        str(msg)
        == "This is an error message.\n\nFor further information visit 'https://distilabel.argilla.io/latest/sections/getting_started/faq/'"
    )
