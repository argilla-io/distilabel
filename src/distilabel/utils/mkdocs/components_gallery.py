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
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import pandas as pd
from jinja2 import Template
from mkdocs.config.base import Config
from mkdocs.config.config_options import Type
from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File
from mkdocs_section_index import SectionPage

from distilabel.utils.export_components_info import export_components_info

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.nav import Navigation

_COMPONENTS_LIST_TEMPLATE = Template(
    open(
        str(
            importlib_resources.files("distilabel")
            / "utils"
            / "mkdocs"
            / "templates"
            / "components-gallery"
            / "components-list.jinja2"
        )
    ).read(),
)

_STEP_DETAIL_TEMPLATE = Template(
    open(
        str(
            importlib_resources.files("distilabel")
            / "utils"
            / "mkdocs"
            / "templates"
            / "components-gallery"
            / "step-detail.jinja2"
        )
    ).read(),
)

_LLM_DETAIL_TEMPLATE = Template(
    open(
        str(
            importlib_resources.files("distilabel")
            / "utils"
            / "mkdocs"
            / "templates"
            / "components-gallery"
            / "llm-detail.jinja2"
        )
    ).read()
)

_STEPS_CATEGORY_TO_ICON = {
    "critique": ":material-comment-edit:",
    "embedding": ":material-vector-line:",
    "evol": ":material-dna:",
    "filtering": ":material-filter:",
    "format": ":material-format-list-bulleted:",
    "load": ":material-file-download:",
    "preference": ":material-poll:",
    "save": ":material-content-save:",
    "scorer": ":octicons-number-16:",
    "text-generation": ":material-text-box-edit:",
    "text-manipulation": ":material-receipt-text-edit:",
    "columns": ":material-table-column:",
}

_STEP_CATEGORY_TO_DESCRIPTION = {
    "text-generation": "Text generation steps are used to generate text based on a given prompt.",
    "evol": "Evol steps are used to rewrite input text and evolve it to a higher quality.",
    "text-manipulation": "Text manipulation steps are used to manipulate or rewrite an input text.",
    "critique": "Critique steps are used to provide feedback on the quality of the data with a written explanation.",
    "scorer": "Scorer steps are used to evaluate and score the data with a numerical value.",
    "preference": "Preference steps are used to collect preferences on the data with numerical values or ranks.",
    "embedding": "Embedding steps are used to generate embeddings for the data.",
    "columns": "Columns steps are used to manipulate columns in the data.",
    "filtering": "Filtering steps are used to filter the data based on some criteria.",
    "format": "Format steps are used to format the data.",
    "load": "Load steps are used to load the data.",
    "save": "Save steps are used to save the data.",
}


class ComponentsGalleryConfig(Config):
    enabled = Type(bool, default=True)
    page_title = Type(str, default="Components Gallery")
    add_after_page = Type(str, default=None)


class ComponentsGalleryPlugin(BasePlugin[ComponentsGalleryConfig]):
    """A MkDocs plugin to generate a components gallery page for `distilabel` components.

    Attributes:
        file_paths: A dictionary to store the paths of the generated files. The keys are
            the subsections of the gallery and the values are the paths of the files.
    """

    def __init__(self) -> None:
        super().__init__()

        self.file_paths = {}

    def on_config(self, config: "MkDocsConfig") -> Union["MkDocsConfig", None]:
        if not self.config.enabled:
            return

    def on_files(
        self, files: "Files", *, config: "MkDocsConfig"
    ) -> Union["Files", None]:
        """Generates the files for the components gallery automatically from the docstrings.

        Args:
            files: The files collection.
            config: The MkDocs configuration.

        Returns:
            The files collection with the new files added.
        """
        src_dir = Path(config["site_dir"])

        components_info = export_components_info()

        # Generate the `components-gallery/index.md`
        self.file_paths["components_gallery"] = self._generate_component_gallery_index(
            src_dir=src_dir
        )

        # Create and write content to subsections
        self.file_paths["steps"] = self._generate_steps_pages(
            src_dir=src_dir, steps=components_info["steps"]
        )
        self.file_paths["tasks"] = self._generate_tasks_pages(
            src_dir=src_dir, tasks=components_info["tasks"]
        )
        self.file_paths["llms"] = self._generate_llms_pages(
            src_dir=src_dir, llms=components_info["llms"]
        )
        self.file_paths["embeddings"] = self._generate_embeddings_pages(
            src_dir=src_dir, embeddings=components_info["embeddings"]
        )

        # Add the new files to the files collections
        for relative_file_path in [
            self.file_paths["components_gallery"],
            *self.file_paths["steps"],
            *self.file_paths["tasks"],
            *self.file_paths["llms"],
            *self.file_paths["embeddings"],
        ]:
            file = File(
                path=relative_file_path,
                src_dir=str(src_dir),
                dest_dir=config.site_dir,
                use_directory_urls=config.use_directory_urls,
            )
            file.generated_by = "distilabel/components-gallery"  # type: ignore
            files.append(file)

        return files

    def _generate_component_gallery_index(self, src_dir: Path) -> str:
        """Generates the `components-gallery/index.md` file.

        Args:
            src_dir: The path to the source directory.

        Returns:
            The relative path to the generated file.
        """
        index_template_path = str(
            importlib_resources.files("distilabel")
            / "utils"
            / "mkdocs"
            / "templates"
            / "components-gallery"
            / "index.md"
        )

        with open(index_template_path) as f:
            index_template = f.read()

        components_gallery_path_relative = "components-gallery/index.md"
        components_gallery_path = src_dir / components_gallery_path_relative
        components_gallery_path.parent.mkdir(parents=True, exist_ok=True)
        with open(components_gallery_path, "w") as f:
            f.write(index_template)

        return components_gallery_path_relative

    def _generate_steps_pages(self, src_dir: Path, steps: list) -> List[str]:
        """Generates the files for the `Steps` subsection of the components gallery.

        Args:
            src_dir: The path to the source directory.
            steps: The list of `Step` components.

        Returns:
            The relative paths to the generated files.
        """

        paths = ["components-gallery/steps/index.md"]
        steps_gallery_page_path = src_dir / paths[0]
        steps_gallery_page_path.parent.mkdir(parents=True, exist_ok=True)

        # Create detail page for each `Step`
        for step in steps:
            docstring = step["docstring"]
            if docstring["icon"] == "" and docstring["categories"]:
                first_category = docstring["categories"][0]
                docstring["icon"] = _STEPS_CATEGORY_TO_ICON.get(first_category, "")

            name = step["name"]

            content = _STEP_DETAIL_TEMPLATE.render(
                step=step,
                mermaid_diagram=_generate_mermaid_diagram_for_io(
                    step_name=step["name"],
                    inputs=list(docstring["input_columns"].keys()),
                    outputs=list(docstring["output_columns"].keys()),
                ),
            )

            step_path = f"components-gallery/steps/{name.lower()}.md"
            path = src_dir / step_path
            with open(path, "w") as f:
                f.write(content)

            paths.append(step_path)

        # Create the `components-gallery/steps.md` file
        content = _COMPONENTS_LIST_TEMPLATE.render(
            title="Steps Gallery",
            description="",
            components=steps,
            default_icon=":material-step-forward:",
        )

        with open(steps_gallery_page_path, "w") as f:
            f.write(content)

        return paths

    def _generate_tasks_pages(self, src_dir: Path, tasks: list) -> List[str]:
        """Generates the files for the `Tasks` subsection of the components gallery.

        Args:
            src_dir: The path to the source directory.
            tasks: The list of `Task` components.

        Returns:
            The relative paths to the generated files.
        """

        paths = ["components-gallery/tasks/index.md"]
        tasks_gallery_page_path = src_dir / paths[0]
        tasks_gallery_page_path.parent.mkdir(parents=True, exist_ok=True)

        # Create detail page for each `Task`
        for task in tasks:
            docstring = task["docstring"]
            if docstring["icon"] == "" and docstring["categories"]:
                first_category = docstring["categories"][0]
                docstring["icon"] = _STEPS_CATEGORY_TO_ICON.get(first_category, "")

            name = task["name"]

            content = _STEP_DETAIL_TEMPLATE.render(
                step=task,
                mermaid_diagram=_generate_mermaid_diagram_for_io(
                    step_name=task["name"],
                    inputs=list(docstring["input_columns"].keys()),
                    outputs=list(docstring["output_columns"].keys()),
                ),
            )

            task_path = f"components-gallery/tasks/{name.lower()}.md"
            path = src_dir / task_path
            with open(path, "w") as f:
                f.write(content)

            paths.append(task_path)

        global _STEP_CATEGORY_TO_DESCRIPTION
        categories = list(_STEP_CATEGORY_TO_DESCRIPTION.keys())
        table = pd.DataFrame(
            {
                "Category": categories,
                "Icon": [_STEPS_CATEGORY_TO_ICON[category] for category in categories],
                "Description": [
                    _STEP_CATEGORY_TO_DESCRIPTION[category] for category in categories
                ],
            }
        ).to_markdown(index=False)

        description = [
            '??? info "Task Category Overview"',
            "    The tasks gallery page showcases the different types of tasks that can be performed with `distilabel`.",
            "",
        ]
        for row in table.split("\n"):
            description.append(f"    {row}")

        # Create the `components-gallery/steps/index.md` file
        content = _COMPONENTS_LIST_TEMPLATE.render(
            title="Tasks Gallery",
            description="\n".join(description),
            components=tasks,
            default_icon=":material-check-outline:",
        )

        with open(tasks_gallery_page_path, "w") as f:
            f.write(content)

        return paths

    def _generate_llms_pages(self, src_dir: Path, llms: list) -> List[str]:
        """Generates the files for the `LLMs` subsection of the components gallery.

        Args:
            src_dir: The path to the source directory.
            llms: The list of `LLM` components.

        Returns:
            The relative paths to the generated files.
        """

        paths = ["components-gallery/llms/index.md"]
        steps_gallery_page_path = src_dir / paths[0]
        steps_gallery_page_path.parent.mkdir(parents=True, exist_ok=True)

        # Create detail page for each `LLM`
        for llm in llms:
            content = _LLM_DETAIL_TEMPLATE.render(llm=llm)

            llm_path = f"components-gallery/llms/{llm['name'].lower()}.md"
            path = src_dir / llm_path
            with open(path, "w") as f:
                f.write(content)

            paths.append(llm_path)

        # Create the `components-gallery/llms/index.md` file
        content = _COMPONENTS_LIST_TEMPLATE.render(
            title="LLMs Gallery",
            description="",
            components=llms,
            component_group="llms",
            default_icon=":material-brain:",
        )

        with open(steps_gallery_page_path, "w") as f:
            f.write(content)

        return paths

    def _generate_embeddings_pages(self, src_dir: Path, embeddings: list) -> List[str]:
        """Generates the files for the `Embeddings` subsection of the components gallery.

        Args:
            src_dir: The path to the source directory.
            embeddings: The list of `Embeddings` components.

        Returns:
            The relative paths to the generated files.
        """

        paths = ["components-gallery/embeddings/index.md"]
        steps_gallery_page_path = src_dir / paths[0]
        steps_gallery_page_path.parent.mkdir(parents=True, exist_ok=True)

        # Create detail page for each `LLM`
        for embeddings_model in embeddings:
            content = _LLM_DETAIL_TEMPLATE.render(llm=embeddings_model)

            llm_path = (
                f"components-gallery/embeddings/{embeddings_model['name'].lower()}.md"
            )
            path = src_dir / llm_path
            with open(path, "w") as f:
                f.write(content)

            paths.append(llm_path)

        # Create the `components-gallery/llms/index.md` file
        content = _COMPONENTS_LIST_TEMPLATE.render(
            title="Embeddings Gallery",
            description="",
            components=embeddings,
            component_group="embeddings",
            default_icon=":material-vector-line:",
        )

        with open(steps_gallery_page_path, "w") as f:
            f.write(content)

        return paths

    def on_nav(
        self, nav: "Navigation", *, config: "MkDocsConfig", files: "Files"
    ) -> Union["Navigation", None]:
        """Adds the components gallery to the navigation bar.

        Args:
            nav: The navigation bar.
            config: The MkDocs configuration.
            files: The files collection.

        Returns:
            The navigation bar with the components gallery added.
        """
        # Find the files in the files collection
        components_gallery_file = files.get_file_from_path(
            self.file_paths["components_gallery"]
        )
        steps_file = files.get_file_from_path(self.file_paths["steps"][0])
        tasks_file = files.get_file_from_path(self.file_paths["tasks"][0])
        llms_file = files.get_file_from_path(self.file_paths["llms"][0])
        steps_files = [
            files.get_file_from_path(path) for path in self.file_paths["steps"][0:]
        ]
        tasks_files = [
            files.get_file_from_path(path) for path in self.file_paths["tasks"][0:]
        ]
        llms_files = [
            files.get_file_from_path(path) for path in self.file_paths["llms"][0:]
        ]

        # Create subsections
        steps_page = SectionPage(
            "Steps", file=steps_file, config=config, children=steps_files
        )  # type: ignore
        tasks_page = SectionPage(
            "Tasks", file=tasks_file, config=config, children=tasks_files
        )  # type: ignore
        llms_page = SectionPage(
            "LLMs", file=llms_file, config=config, children=llms_files
        )  # type: ignore

        # Create the gallery section
        page = SectionPage(
            title=self.config.page_title,
            file=components_gallery_file,
            config=config,
            children=[steps_page, tasks_page, llms_page],
        )

        # Add the page
        nav.pages.append(page)

        # Add the page to the navigation bar
        if self.config.add_after_page:
            for i, item in enumerate(nav.items):
                if item.title == self.config.add_after_page:
                    nav.items.insert(i + 1, page)
                    break
        else:
            nav.items.append(page)

        return nav


def _generate_mermaid_diagram_for_io(  # noqa: C901
    step_name: str, inputs: List[str], outputs: List[str]
) -> str:
    """Generates a mermaid diagram for representing the input and output columns of a `Step`.

    Args:
        step_name: The name of the `Step`.
        inputs: The input columns of the `Step`.
        outputs: The output columns of the `Step`.

    Returns:
        The mermaid diagram syntax representing the input and output columns of the `Step`.
    """
    # Initialize the mermaid diagram syntax
    mermaid = "graph TD\n"

    # Add dataset columns (inputs and outputs)
    mermaid += "\tsubgraph Dataset\n"
    if inputs:
        mermaid += "\t\tsubgraph Columns\n"
        for i, col in enumerate(inputs):
            mermaid += f"\t\t\tICOL{i}[{col}]\n"
        mermaid += "\t\tend\n"

    if outputs:
        mermaid += "\t\tsubgraph New columns\n"
        for i, col in enumerate(outputs):
            mermaid += f"\t\t\tOCOL{i}[{col}]\n"
        mermaid += "\t\tend\n"
    mermaid += "\tend\n\n"

    # Add steps
    mermaid += f"\tsubgraph {step_name}\n"
    if inputs:
        input_cols = ", ".join(inputs)
        mermaid += f"\t\tStepInput[Input Columns: {input_cols}]\n"

    if outputs:
        output_cols = ", ".join(outputs)
        mermaid += f"\t\tStepOutput[Output Columns: {output_cols}]\n"

    mermaid += "\tend\n\n"

    # Add connections
    if inputs:
        for i in range(len(inputs)):
            mermaid += f"\tICOL{i} --> StepInput\n"

    if outputs:
        for i in range(len(outputs)):
            mermaid += f"\tStepOutput --> OCOL{i}\n"

    if inputs and outputs:
        mermaid += "\tStepInput --> StepOutput\n"

    return mermaid
