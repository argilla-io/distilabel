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


def test_imports() -> None:
    # ruff: noqa
    from distilabel.llms import (
        AnthropicLLM,
        AnyscaleLLM,
        AsyncLLM,
        AzureOpenAILLM,
        CudaDevicePlacementMixin,
        GenerateOutput,
        HiddenState,
        InferenceEndpointsLLM,
        MixtureOfAgentsLLM,
        LlamaCppLLM,
        LLM,
        LiteLLM,
        MistralLLM,
        OpenAILLM,
        TogetherLLM,
        TransformersLLM,
        VertexAILLM,
        vLLM,
    )

    from distilabel.pipeline import Pipeline

    from distilabel.steps import (
        StepResources,
        CombineColumns,
        GroupColumns,
        MergeColumns,
        ConversationTemplate,
        DeitaFiltering,
        ExpandColumns,
        FormatChatGenerationDPO,
        FormatChatGenerationSFT,
        FormatTextGenerationDPO,
        FormatTextGenerationSFT,
        GeneratorStep,
        GlobalStep,
        GeneratorStepOutput,
        KeepColumns,
        LoadDataFromDicts,
        LoadDataFromHub,
        LoadDataFromDisk,
        PushToHub,
        Step,
        StepOutput,
        PreferenceToArgilla,
        TextGenerationToArgilla,
        step,
    )

    from distilabel.steps.tasks import (
        Task,
        GeneratorTask,
        ChatItem,
        ChatType,
        ComplexityScorer,
        EvolInstruct,
        EvolComplexity,
        EvolComplexityGenerator,
        EvolInstructGenerator,
        GenerateEmbeddings,
        Genstruct,
        BitextRetrievalGenerator,
        EmbeddingTaskGenerator,
        GenerateLongTextMatchingData,
        GenerateShortTextMatchingData,
        GenerateTextClassificationData,
        GenerateTextRetrievalData,
        MonolingualTripletGenerator,
        InstructionBacktranslation,
        PairRM,
        PrometheusEval,
        QualityScorer,
        SelfInstruct,
        StructuredGeneration,
        TextGeneration,
        UltraFeedback,
    )
