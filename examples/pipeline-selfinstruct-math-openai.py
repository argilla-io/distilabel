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

import os

from distilabel.tasks import SelfInstructTask
from distilabel.pipeline import Pipeline
from distilabel.llm import OpenAILLM

from datasets import Dataset

math_topics = [
    "Algebraic Expressions",
    "Linear Equations",
    "Quadratic Equations",
    "Polynomial Functions",
    "Rational Expressions",
    "Exponential Functions",
    "Logarithmic Functions",
    "Sequences and Series",
    "Matrices",
    "Determinants",
    "Complex Numbers",
    "Trigonometry",
    "Geometry",
    "Coordinate Geometry",
    "Vector Algebra",
    "Statistics",
    "Probability",
    "Calculus",
    "Differential Calculus",
    "Integral Calculus",
    "Limits and Continuity",
    "Differentiation",
    "Integration",
    "Theorems of Calculus",
    "Mathematical Reasoning",
    "Set Theory",
    "Number Theory",
    "Permutations and Combinations",
    "Binomial Theorem",
    "Arithmetic Progressions",
    "Geometric Progressions",
    "Harmonic Progressions",
    "Trigonometric Ratios",
    "Trigonometric Identities",
    "Inverse Trigonometric Functions",
    "Hyperbolic Functions",
    "Conic Sections",
    "Circle Geometry",
    "Ellipse Geometry",
    "Parabola Geometry",
    "Hyperbola Geometry",
    "Function Theory",
    "Graph Theory",
    "Differential Equations",
    "Mathematical Induction",
    "Discrete Mathematics",
    "Linear Programming",
    "Analytical Geometry",
    "Euclidean Geometry",
    "Non-Euclidean Geometry"
]

dataset = Dataset.from_dict({
    "input": math_topics
})

instruction_task = SelfInstructTask(
    application_description="A question-answering assistant for engaging and challenging math quizzes and problems"
)

instruction_generator = OpenAILLM(
    task=instruction_task,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    num_threads=4,
    max_new_tokens=1024
)

pipeline = Pipeline(
    generator=instruction_generator
)

distiset = pipeline.generate(dataset=dataset, num_generations=4, batch_size=2)

instructions = []
for generations in distiset["generations"]:
    for generation in generations:
        instructions.extend(generation)
print(f"Number of generated instructions: {len(instructions)}")
print(instructions)

# Output:
# Number of generated instructions: 2044
# 1. Provide an explanation for solving a quadratic equation step by step.
# 2. What is the process for simplifying an algebraic expression with exponents?
# 3. Detail how to factorize a polynomial equation.
# 4. How can one determine the maximum or minimum value of a quadratic function?
# 5. Explain the concept of inequalities and how to solve them algebraically.
# 6. Describe the procedure for finding the roots of a cubic equation.
# 7. What are the different types of factoring techniques used in algebra?
# 8. Can you outline the steps for evaluating an algebraic expression using substitution?
# 9. Compare and contrast linear and quadratic equations in terms of their solutions and graphs.
# 10. How can one determine if a given graph represents a linear or quadratic equation?
# 1. How can I simplify the algebraic expression (x^2 + 3x + 2)(2x - 1)?
# 2. Provide step-by-step instructions on how to solve the equation 4(x + 2) - 3 = 7(2x - 1).
# 3. What is the value of x in the equation 3(x - 4) = 5x + 6?
# 4. Detail the process of factoring the expression 12x^2 - 7x - 10.
# 5. What is the result of expanding the binomial (2x - 3)^2?

