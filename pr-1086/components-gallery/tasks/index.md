---
hide:
  - toc
  - navigation
---
# Tasks Gallery

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
    | :label:                         | labelling           | Labelling steps are used to label the data.                                                        |

<div class="grid cards" markdown>


-   :material-text-box-edit:{ .lg .middle } __APIGenGenerator__

    ---

    Generate queries and answers for the given functions in JSON format.

    [:octicons-arrow-right-24: APIGenGenerator](apigengenerator.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __Genstruct__

    ---

    Generate a pair of instruction-response from a document using an `LLM`.

    [:octicons-arrow-right-24: Genstruct](genstruct.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __Magpie__

    ---

    Generates conversations using an instruct fine-tuned LLM.

    [:octicons-arrow-right-24: Magpie](magpie.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __MathShepherdCompleter__

    ---

    Math Shepherd Completer and auto-labeller task.

    [:octicons-arrow-right-24: MathShepherdCompleter](mathshepherdcompleter.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __MathShepherdGenerator__

    ---

    Math Shepherd solution generator.

    [:octicons-arrow-right-24: MathShepherdGenerator](mathshepherdgenerator.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __SelfInstruct__

    ---

    Generate instructions based on a given input using an `LLM`.

    [:octicons-arrow-right-24: SelfInstruct](selfinstruct.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __TextGeneration__

    ---

    Text generation with an `LLM` given a prompt.

    [:octicons-arrow-right-24: TextGeneration](textgeneration.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __TextGenerationWithImage__

    ---

    Text generation with images with an `LLM` given a prompt.

    [:octicons-arrow-right-24: TextGenerationWithImage](textgenerationwithimage.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __URIAL__

    ---

    Generates a response using a non-instruct fine-tuned model.

    [:octicons-arrow-right-24: URIAL](urial.md){ .bottom }

-   :material-text-box-edit:{ .lg .middle } __MagpieGenerator__

    ---

    Generator task the generates instructions or conversations using Magpie.

    [:octicons-arrow-right-24: MagpieGenerator](magpiegenerator.md){ .bottom }

-   :material-chat:{ .lg .middle } __ChatGeneration__

    ---

    Generates text based on a conversation.

    [:octicons-arrow-right-24: ChatGeneration](chatgeneration.md){ .bottom }

-   :material-label:{ .lg .middle } __ArgillaLabeller__

    ---

    Annotate Argilla records based on input fields, example records and question settings.

    [:octicons-arrow-right-24: ArgillaLabeller](argillalabeller.md){ .bottom }

-   :material-label:{ .lg .middle } __TextClassification__

    ---

    Classifies text into one or more categories or labels.

    [:octicons-arrow-right-24: TextClassification](textclassification.md){ .bottom }

-   :material-dna:{ .lg .middle } __EvolInstruct__

    ---

    Evolve instructions using an `LLM`.

    [:octicons-arrow-right-24: EvolInstruct](evolinstruct.md){ .bottom }

-   :material-dna:{ .lg .middle } __EvolComplexity__

    ---

    Evolve instructions to make them more complex using an `LLM`.

    [:octicons-arrow-right-24: EvolComplexity](evolcomplexity.md){ .bottom }

-   :material-dna:{ .lg .middle } __EvolQuality__

    ---

    Evolve the quality of the responses using an `LLM`.

    [:octicons-arrow-right-24: EvolQuality](evolquality.md){ .bottom }

-   :material-dna:{ .lg .middle } __EvolInstructGenerator__

    ---

    Generate evolved instructions using an `LLM`.

    [:octicons-arrow-right-24: EvolInstructGenerator](evolinstructgenerator.md){ .bottom }

-   :material-dna:{ .lg .middle } __EvolComplexityGenerator__

    ---

    Generate evolved instructions with increased complexity using an `LLM`.

    [:octicons-arrow-right-24: EvolComplexityGenerator](evolcomplexitygenerator.md){ .bottom }

-   :material-comment-edit:{ .lg .middle } __InstructionBacktranslation__

    ---

    Self-Alignment with Instruction Backtranslation.

    [:octicons-arrow-right-24: InstructionBacktranslation](instructionbacktranslation.md){ .bottom }

-   :material-comment-edit:{ .lg .middle } __PrometheusEval__

    ---

    Critique and rank the quality of generations from an `LLM` using Prometheus 2.0.

    [:octicons-arrow-right-24: PrometheusEval](prometheuseval.md){ .bottom }

-   :octicons-number-16:{ .lg .middle } __ComplexityScorer__

    ---

    Score instructions based on their complexity using an `LLM`.

    [:octicons-arrow-right-24: ComplexityScorer](complexityscorer.md){ .bottom }

-   :octicons-number-16:{ .lg .middle } __QualityScorer__

    ---

    Score responses based on their quality using an `LLM`.

    [:octicons-arrow-right-24: QualityScorer](qualityscorer.md){ .bottom }

-   :material-poll:{ .lg .middle } __CLAIR__

    ---

    Contrastive Learning from AI Revisions (CLAIR).

    [:octicons-arrow-right-24: CLAIR](clair.md){ .bottom }

-   :material-poll:{ .lg .middle } __UltraFeedback__

    ---

    Rank generations focusing on different aspects using an `LLM`.

    [:octicons-arrow-right-24: UltraFeedback](ultrafeedback.md){ .bottom }

-   :material-poll:{ .lg .middle } __PairRM__

    ---

    Rank the candidates based on the input using the `LLM` model.

    [:octicons-arrow-right-24: PairRM](pairrm.md){ .bottom }

-   :material-vector-line:{ .lg .middle } __GenerateSentencePair__

    ---

    Generate a positive and negative (optionally) sentences given an anchor sentence.

    [:octicons-arrow-right-24: GenerateSentencePair](generatesentencepair.md){ .bottom }

-   :material-vector-line:{ .lg .middle } __GenerateEmbeddings__

    ---

    Generate embeddings using the last hidden state of an `LLM`.

    [:octicons-arrow-right-24: GenerateEmbeddings](generateembeddings.md){ .bottom }

-   :material-scatter-plot:{ .lg .middle } __TextClustering__

    ---

    Task that clusters a set of texts and generates summary labels for each cluster.

    [:octicons-arrow-right-24: TextClustering](textclustering.md){ .bottom }

-   :material-scatter-plot:{ .lg .middle } __TextClustering__

    ---

    Task that clusters a set of texts and generates summary labels for each cluster.

    [:octicons-arrow-right-24: TextClustering](textclustering.md){ .bottom }

-   :material-filter:{ .lg .middle } __APIGenSemanticChecker__

    ---

    Generate queries and answers for the given functions in JSON format.

    [:octicons-arrow-right-24: APIGenSemanticChecker](apigensemanticchecker.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __GenerateTextRetrievalData__

    ---

    Generate text retrieval data with an `LLM` to later on train an embedding model.

    [:octicons-arrow-right-24: GenerateTextRetrievalData](generatetextretrievaldata.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __GenerateShortTextMatchingData__

    ---

    Generate short text matching data with an `LLM` to later on train an embedding model.

    [:octicons-arrow-right-24: GenerateShortTextMatchingData](generateshorttextmatchingdata.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __GenerateLongTextMatchingData__

    ---

    Generate long text matching data with an `LLM` to later on train an embedding model.

    [:octicons-arrow-right-24: GenerateLongTextMatchingData](generatelongtextmatchingdata.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __GenerateTextClassificationData__

    ---

    Generate text classification data with an `LLM` to later on train an embedding model.

    [:octicons-arrow-right-24: GenerateTextClassificationData](generatetextclassificationdata.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __StructuredGeneration__

    ---

    Generate structured content for a given `instruction` using an `LLM`.

    [:octicons-arrow-right-24: StructuredGeneration](structuredgeneration.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __MonolingualTripletGenerator__

    ---

    Generate monolingual triplets with an `LLM` to later on train an embedding model.

    [:octicons-arrow-right-24: MonolingualTripletGenerator](monolingualtripletgenerator.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __BitextRetrievalGenerator__

    ---

    Generate bitext retrieval data with an `LLM` to later on train an embedding model.

    [:octicons-arrow-right-24: BitextRetrievalGenerator](bitextretrievalgenerator.md){ .bottom }

-   :material-check-outline:{ .lg .middle } __EmbeddingTaskGenerator__

    ---

    Generate task descriptions for embedding-related tasks using an `LLM`.

    [:octicons-arrow-right-24: EmbeddingTaskGenerator](embeddingtaskgenerator.md){ .bottom }


</div>