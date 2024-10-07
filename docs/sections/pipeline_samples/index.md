---
hide: toc
---
# Tutorials

- **End-to-end tutorials** provide detailed step-by-step explanations and the code used for end-to-end workflows.
- **Paper implementations** provide reproductions of fundamental papers in the synthetic data domain.
- **Examples** don't provide explenations but simply show code for different tasks.

## End-to-end tutorials

<div class="grid cards" markdown>

-   __Generate a preference dataset__

    ---

    Learn about synthetic data generation for ORPO and DPO.

    [:octicons-arrow-right-24: Tutorial](tutorials/generate_preference_dataset.ipynb)


-   __Clean an existing preference dataset__

    ---

    Learn about how to provide AI feedback to clean an existing dataset.

    [:octicons-arrow-right-24: Tutorial](tutorials/clean_existing_dataset.ipynb)


-   __Retrieval and reranking models__

    ---

    Learn about synthetic data generation for fine-tuning custom retrieval and reranking models.

    [:octicons-arrow-right-24: Tutorial](tutorials/GenerateSentencePair.ipynb)

</div>

## Paper Implementations

<div class="grid cards" markdown>

-   __Deepseek Prover__

    ---

    Learn about an approach to generate mathematical proofs for theorems generated from informal math problems.

    [:octicons-arrow-right-24: Example](papers/deepseek_prover.md)

-   __DEITA__

    ---

    Learn about prompt, response tuning for complexity and quality and LLMs as judges for automatic data selection.

    [:octicons-arrow-right-24: Paper](papers/deita.md)

-   __Instruction Backtranslation__

    ---

    Learn about automatically labeling human-written text with corresponding instructions.

    [:octicons-arrow-right-24: Paper](papers/instruction_backtranslation.md)

-   __Prometheus 2__

    ---

    Learn about using open-source models as judges for direct assessment and pair-wise ranking.

    [:octicons-arrow-right-24: Paper](papers/prometheus.md)

-   __UltraFeedback__

    ---

    Learn about a large-scale, fine-grained, diverse preference dataset, used for training powerful reward and critic models.

    [:octicons-arrow-right-24: Paper](papers/ultrafeedback.md)

-   __APIGen__

    ---

    Learn how to create verifiable high-quality datases for function-calling applications.

    [:octicons-arrow-right-24: Paper](papers/apigen.md)

-   __CLAIR__

    ---

    Learn Contrastive Learning from AI Revisions (CLAIR), a data-creation method which leads to more contrastive preference pairs.

    [:octicons-arrow-right-24: Paper](papers/clair.md)

</div>

## Examples

<div class="grid cards" markdown>

-   __Benchmarking with distilabel__

    ---

    Learn about reproducing the Arena Hard benchmark with disitlabel.

    [:octicons-arrow-right-24: Example](examples/benchmarking_with_distilabel.md)

-   __Structured generation with outlines__

    ---

    Learn about generating RPG characters following a pydantic.BaseModel with outlines in distilabel.

    [:octicons-arrow-right-24: Example](examples/llama_cpp_with_outlines.md)

-   __Structured generation with instructor__

    ---

    Learn about answering instructions with knowledge graphs defined as pydantic.BaseModel objects using instructor in distilabel.

    [:octicons-arrow-right-24: Example](examples/mistralai_with_instructor.md)

-   __Create a social network with FinePersonas__

    ---

    Learn how to leverage FinePersonas to create a synthetic social network and fine-tune adapters for Multi-LoRA.

    [:octicons-arrow-right-24: Example](examples/fine_personas_social_network.md)


</div>





