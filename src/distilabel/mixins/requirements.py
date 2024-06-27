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

import importlib.util
from importlib.metadata import version
from typing import List, Union

from packaging.requirements import InvalidRequirement, Requirement


class RequirementsMixin:
    """Mixin for classes that have `requirements` attribute.

    Used to add requirements to a `Step` and a `Pipeline`.
    """

    _requirements: Union[List[Requirement], None] = []

    def _gather_requirements(self) -> List[str]:
        """This method will be overwritten in the `BasePipeline` class to gather the requirements
        from each step.
        """
        return []

    @property
    def requirements(self) -> List[str]:
        """Return a list of requirements that must be installed to run the `Pipeline`.

        The requirements in a Pipeline will include the requirements from all the steps (if any).

        Returns:
            List of requirements that must be installed to run the `Pipeline`, sorted alphabetically.
        """
        self.requirements = self._gather_requirements()

        return [str(r) for r in self._requirements]

    @requirements.setter
    def requirements(self, _requirements: List[str]) -> None:
        requirements = []
        if not isinstance(_requirements, list):
            _requirements = [_requirements]

        for r in _requirements:
            try:
                requirements.append(Requirement(r))
            except InvalidRequirement:
                self._logger.warning(f"Invalid requirement: `{r}`")

        self._requirements = sorted(
            set(self._requirements).union(set(requirements)), key=lambda x: str(x)
        )

    def requirements_to_install(self) -> List[str]:
        """Check if the requirements are installed in the current environment, and returns the ones that aren't.

        Returns:
            List of requirements required to run the pipeline that are not installed in the current environment.
        """

        to_install = []
        for req in self.requirements:
            requirement = Requirement(req)
            if importlib.util.find_spec(requirement.name):
                if (str(requirement.specifier) != "") and (
                    version(requirement.name) != str(requirement.specifier)
                ):
                    to_install.append(req)
            else:
                to_install.append(req)
        return to_install
