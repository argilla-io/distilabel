---
hide:
  - toc
  - navigation
---
# Steps Gallery



<div class="grid cards" markdown>


-   :material-filter:{ .lg .middle } __DeitaFiltering__

    ---

    Filter dataset rows using DEITA filtering strategy.

    [:octicons-arrow-right-24: DeitaFiltering](deitafiltering.md){ .bottom }

-   :material-vector-line:{ .lg .middle } __FaissNearestNeighbour__

    ---

    Create a `faiss` index to get the nearest neighbours.

    [:octicons-arrow-right-24: FaissNearestNeighbour](faissnearestneighbour.md){ .bottom }

-   :material-filter:{ .lg .middle } __EmbeddingDedup__

    ---

    Deduplicates text using embeddings.

    [:octicons-arrow-right-24: EmbeddingDedup](embeddingdedup.md){ .bottom }

-   :material-content-save:{ .lg .middle } __PushToHub__

    ---

    Push data to a Hugging Face Hub dataset.

    [:octicons-arrow-right-24: PushToHub](pushtohub.md){ .bottom }

-   :material-step-forward:{ .lg .middle } __PreferenceToArgilla__

    ---

    Creates a preference dataset in Argilla.

    [:octicons-arrow-right-24: PreferenceToArgilla](preferencetoargilla.md){ .bottom }

-   :material-step-forward:{ .lg .middle } __TextGenerationToArgilla__

    ---

    Creates a text generation dataset in Argilla.

    [:octicons-arrow-right-24: TextGenerationToArgilla](textgenerationtoargilla.md){ .bottom }

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

-   :material-step-forward:{ .lg .middle } __CombineColumns__

    ---

    `CombineColumns` is deprecated and will be removed in version 1.5.0, use `GroupColumns` instead.

    [:octicons-arrow-right-24: CombineColumns](combinecolumns.md){ .bottom }

-   :material-table-column:{ .lg .middle } __KeepColumns__

    ---

    Keeps selected columns in the dataset.

    [:octicons-arrow-right-24: KeepColumns](keepcolumns.md){ .bottom }

-   :material-table-column:{ .lg .middle } __MergeColumns__

    ---

    Merge columns from a row.

    [:octicons-arrow-right-24: MergeColumns](mergecolumns.md){ .bottom }

-   :material-vector-line:{ .lg .middle } __EmbeddingGeneration__

    ---

    Generate embeddings using an `Embeddings` model.

    [:octicons-arrow-right-24: EmbeddingGeneration](embeddinggeneration.md){ .bottom }

-   :material-filter:{ .lg .middle } __MinHashDedup__

    ---

    Deduplicates text using `MinHash` and `MinHashLSH`.

    [:octicons-arrow-right-24: MinHashDedup](minhashdedup.md){ .bottom }

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

-   :octicons-number-16:{ .lg .middle } __RewardModelScore__

    ---

    Assign a score to a response using a Reward Model.

    [:octicons-arrow-right-24: RewardModelScore](rewardmodelscore.md){ .bottom }

-   :material-receipt-text-edit:{ .lg .middle } __TruncateTextColumn__

    ---

    Truncate a row using a tokenizer or the number of characters.

    [:octicons-arrow-right-24: TruncateTextColumn](truncatetextcolumn.md){ .bottom }

-   :material-file-download:{ .lg .middle } __LoadDataFromDicts__

    ---

    Loads a dataset from a list of dictionaries.

    [:octicons-arrow-right-24: LoadDataFromDicts](loaddatafromdicts.md){ .bottom }

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


</div>