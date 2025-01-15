---
hide:
  - toc
  - navigation
---
# Steps Gallery

??? info "Category Overview"
    The gallery page showcases the different types of components within `distilabel`.

    | Icon                            | Category            | Description                                                                                        |
    |:--------------------------------|:--------------------|:---------------------------------------------------------------------------------------------------|
    | :material-text-box-edit:        | text-generation     | Text generation steps are used to generate text based on a given prompt.                           |
    | :material-chat:                 | chat-generation     | Chat generation steps are used to generate text based on a conversation.                           |
    | :material-label:                | text-classification | Text classification steps are used to classify text into a category.                               |
    | :material-receipt-text-edit:    | text-manipulation   | Text manipulation steps are used to manipulate or rewrite an input text.                           |
    | :material-dna:                  | evol                | Evol steps are used to rewrite input text and evolve it to a higher quality.                       |
    | :material-comment-edit:         | critique            | Critique steps are used to provide feedback on the quality of the data with a written explanation. |
    | :octicons-number-16:            | scorer              | Scorer steps are used to evaluate and score the data with a numerical value.                       |
    | :material-poll:                 | preference          | Preference steps are used to collect preferences on the data with numerical values or ranks.       |
    | :material-vector-line:          | embedding           | Embedding steps are used to generate embeddings for the data.                                      |
    | :material-scatter-plot:         | clustering          | Clustering steps are used to group similar data points together.                                   |
    | :material-table-column:         | columns             | Columns steps are used to manipulate columns in the data.                                          |
    | :material-filter:               | filtering           | Filtering steps are used to filter the data based on some criteria.                                |
    | :material-format-list-bulleted: | format              | Format steps are used to format the data.                                                          |
    | :material-file-download:        | load                | Load steps are used to load the data.                                                              |
    | :octicons-code-16:              | execution           | Executes python functions.                                                                         |
    | :material-content-save:         | save                | Save steps are used to save the data.                                                              |
    | :material-image:                | image-generation    | Image generation steps are used to generate images based on a given prompt.                        |
    | :label:                         | labelling           | Labelling steps are used to label the data.                                                        |

<div class="grid cards" markdown>


-   :material-step-forward:{ .lg .middle } __PreferenceToArgilla__

    ---

    Creates a preference dataset in Argilla.

    [:octicons-arrow-right-24: PreferenceToArgilla](preferencetoargilla.md){ .bottom }

-   :material-step-forward:{ .lg .middle } __TextGenerationToArgilla__

    ---

    Creates a text generation dataset in Argilla.

    [:octicons-arrow-right-24: TextGenerationToArgilla](textgenerationtoargilla.md){ .bottom }

-   :material-step-forward:{ .lg .middle } __CombineColumns__

    ---

    `CombineColumns` is deprecated and will be removed in version 1.5.0, use `GroupColumns` instead.

    [:octicons-arrow-right-24: CombineColumns](combinecolumns.md){ .bottom }

-   :material-content-save:{ .lg .middle } __PushToHub__

    ---

    Push data to a Hugging Face Hub dataset.

    [:octicons-arrow-right-24: PushToHub](pushtohub.md){ .bottom }

-   :material-file-download:{ .lg .middle } __LoadDataFromDicts__

    ---

    Loads a dataset from a list of dictionaries.

    [:octicons-arrow-right-24: LoadDataFromDicts](loaddatafromdicts.md){ .bottom }

-   :material-file-download:{ .lg .middle } __DataSampler__

    ---

    Step to sample from a dataset.

    [:octicons-arrow-right-24: DataSampler](datasampler.md){ .bottom }

-   :material-file-download:{ .lg .middle } __LoadDataFromHub__

    ---

    Loads a dataset from the Hugging Face Hub.

    [:octicons-arrow-right-24: LoadDataFromHub](loaddatafromhub.md){ .bottom }

-   :material-file-download:{ .lg .middle } __LoadDataFromFileSystem__

    ---

    Loads a dataset from a file in your filesystem.

    [:octicons-arrow-right-24: LoadDataFromFileSystem](loaddatafromfilesystem.md){ .bottom }

-   :material-file-download:{ .lg .middle } __LoadDataFromDisk__

    ---

    Load a dataset that was previously saved to disk.

    [:octicons-arrow-right-24: LoadDataFromDisk](loaddatafromdisk.md){ .bottom }

-   :material-format-list-bulleted:{ .lg .middle } __PrepareExamples__

    ---

    Helper step to create examples from `query` and `answers` pairs used as Few Shots in APIGen.

    [:octicons-arrow-right-24: PrepareExamples](prepareexamples.md){ .bottom }

-   :material-format-list-bulleted:{ .lg .middle } __ConversationTemplate__

    ---

    Generate a conversation template from an instruction and a response.

    [:octicons-arrow-right-24: ConversationTemplate](conversationtemplate.md){ .bottom }

-   :material-format-list-bulleted:{ .lg .middle } __FormatTextGenerationDPO__

    ---

    Format the output of your LLMs for Direct Preference Optimization (DPO).

    [:octicons-arrow-right-24: FormatTextGenerationDPO](formattextgenerationdpo.md){ .bottom }

-   :material-format-list-bulleted:{ .lg .middle } __FormatChatGenerationDPO__

    ---

    Format the output of a combination of a `ChatGeneration` + a preference task for Direct Preference Optimization (DPO).

    [:octicons-arrow-right-24: FormatChatGenerationDPO](formatchatgenerationdpo.md){ .bottom }

-   :material-format-list-bulleted:{ .lg .middle } __FormatTextGenerationSFT__

    ---

    Format the output of a `TextGeneration` task for Supervised Fine-Tuning (SFT).

    [:octicons-arrow-right-24: FormatTextGenerationSFT](formattextgenerationsft.md){ .bottom }

-   :material-format-list-bulleted:{ .lg .middle } __FormatChatGenerationSFT__

    ---

    Format the output of a `ChatGeneration` task for Supervised Fine-Tuning (SFT).

    [:octicons-arrow-right-24: FormatChatGenerationSFT](formatchatgenerationsft.md){ .bottom }

-   :material-filter:{ .lg .middle } __DeitaFiltering__

    ---

    Filter dataset rows using DEITA filtering strategy.

    [:octicons-arrow-right-24: DeitaFiltering](deitafiltering.md){ .bottom }

-   :material-filter:{ .lg .middle } __EmbeddingDedup__

    ---

    Deduplicates text using embeddings.

    [:octicons-arrow-right-24: EmbeddingDedup](embeddingdedup.md){ .bottom }

-   :material-filter:{ .lg .middle } __APIGenExecutionChecker__

    ---

    Executes the generated function calls.

    [:octicons-arrow-right-24: APIGenExecutionChecker](apigenexecutionchecker.md){ .bottom }

-   :material-filter:{ .lg .middle } __MinHashDedup__

    ---

    Deduplicates text using `MinHash` and `MinHashLSH`.

    [:octicons-arrow-right-24: MinHashDedup](minhashdedup.md){ .bottom }

-   :material-table-column:{ .lg .middle } __CombineOutputs__

    ---

    Combine the outputs of several upstream steps.

    [:octicons-arrow-right-24: CombineOutputs](combineoutputs.md){ .bottom }

-   :material-table-column:{ .lg .middle } __ExpandColumns__

    ---

    Expand columns that contain lists into multiple rows.

    [:octicons-arrow-right-24: ExpandColumns](expandcolumns.md){ .bottom }

-   :material-table-column:{ .lg .middle } __GroupColumns__

    ---

    Combines columns from a list of `StepInput`.

    [:octicons-arrow-right-24: GroupColumns](groupcolumns.md){ .bottom }

-   :material-table-column:{ .lg .middle } __KeepColumns__

    ---

    Keeps selected columns in the dataset.

    [:octicons-arrow-right-24: KeepColumns](keepcolumns.md){ .bottom }

-   :material-table-column:{ .lg .middle } __MergeColumns__

    ---

    Merge columns from a row.

    [:octicons-arrow-right-24: MergeColumns](mergecolumns.md){ .bottom }

-   :material-scatter-plot:{ .lg .middle } __DBSCAN__

    ---

    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds core

    [:octicons-arrow-right-24: DBSCAN](dbscan.md){ .bottom }

-   :material-scatter-plot:{ .lg .middle } __UMAP__

    ---

    UMAP is a general purpose manifold learning and dimension reduction algorithm.

    [:octicons-arrow-right-24: UMAP](umap.md){ .bottom }

-   :material-vector-line:{ .lg .middle } __FaissNearestNeighbour__

    ---

    Create a `faiss` index to get the nearest neighbours.

    [:octicons-arrow-right-24: FaissNearestNeighbour](faissnearestneighbour.md){ .bottom }

-   :material-vector-line:{ .lg .middle } __EmbeddingGeneration__

    ---

    Generate embeddings using an `Embeddings` model.

    [:octicons-arrow-right-24: EmbeddingGeneration](embeddinggeneration.md){ .bottom }

-   :octicons-number-16:{ .lg .middle } __RewardModelScore__

    ---

    Assign a score to a response using a Reward Model.

    [:octicons-arrow-right-24: RewardModelScore](rewardmodelscore.md){ .bottom }

-   :material-receipt-text-edit:{ .lg .middle } __FormatPRM__

    ---

    Helper step to transform the data into the format expected by the PRM model.

    [:octicons-arrow-right-24: FormatPRM](formatprm.md){ .bottom }

-   :material-receipt-text-edit:{ .lg .middle } __TruncateTextColumn__

    ---

    Truncate a row using a tokenizer or the number of characters.

    [:octicons-arrow-right-24: TruncateTextColumn](truncatetextcolumn.md){ .bottom }


</div>