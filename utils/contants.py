USER = "user"
ASSISTANT = "assistant"

OPEN_AI_MODELS = ['GPT 3', 'GPT 4']
GOOGLE_AI_MODELS = ['Gemini Pro 1.5', 'Gemini Pro 1.0']
ANTHROPIC_AI_MODELS = ['Claude 3 Haiku', 'Claude 3 Sonnet', 'Claude 3 Opus', 'Claude 2']

OPEN_AI_MODELS_NAMES = {
    'GPT 3': 'gpt-3.5-turbo',
    'GPT 4': 'gpt-4-turbo',
}

GOOGLE_AI_MODELS_NAMES = {
    'Gemini Pro 1.5': 'gemini-1.5-pro',
    'Gemini Pro 1.0': 'gemini-1.0-pro',
}

ANTHROPIC_AI_MODELS_NAMES = {
    'Claude 2': 'claude-2.1',
    'Claude 3 Haiku': 'claude-3-haiku-20240307',
    'Claude 3 Opus': 'claude-3-opus-20240229',
    'Claude 3 Sonnet': 'claude-3-sonnet-20240229',
}

MERGE_PROMPT = """

[System]
Please act as an impartial judge and evaluate the responses provided by multiple AI assistants to the user question displayed below. Your task is to combine or synthesize these responses to create a comprehensive and accurate answer. Consider the strengths and weaknesses of each response and aim to produce a final answer that is correct, helpful, and well-rounded. Begin by reviewing the responses from each assistant and identifying key points or information that should be incorporated into the combined answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible.

After reviewing the responses and synthesizing them into a unified answer, provide your final combined response below.

[User Question]
{question}

---

[Assistant A's Answer]
{answer_a}

---

[Assistant B's Answer]
{answer_b}

---

[Assistant C's Answer]
{answer_c}

...

[Assistant N's Answer]
{answer_n}

---

[Combined Answer]
{combined_answer}

"""
PAIRWISE_PROMPT = """

Please act as an impartial judge and evaluate the quality of the responses provided by multiple AI assistants to the 
user question displayed below. Your task is to compare these responses and determine which assistant best follows the 
user’s instructions and answers the user’s question. Your evaluation should consider factors such as the helpfulness, 
relevance, accuracy, depth, creativity, and level of detail of each response. Begin your evaluation by comparing the 
provided responses and provide a short explanation for your decision. Avoid any position biases and ensure that the 
order in which the responses were presented does not influence your decision. Do not favor certain names of the 
assistants. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

[User Question]
{question}

[Assistant A's Answer]
{answer_a}

[Assistant B's Answer]
{answer_b}

[Assistant C's Answer]
{answer_c}

...

[Assistant N's Answer]
{answer_n}

---

**Evaluation:**

To evaluate these responses, consider the following criteria: helpfulness, relevance, accuracy, depth, creativity, 
and level of detail.

- **Assistant A:** [Your evaluation of Assistant A's response]
- **Assistant B:** [Your evaluation of Assistant B's response]
- **Assistant C:** [Your evaluation of Assistant C's response]
- ...
- **Assistant N:** [Your evaluation of Assistant N's response]

Based on your evaluation, output your final verdict by strictly following this format: 
- "[[A]]" if Assistant A is the best,
- "[[B]]" if Assistant B is the best,
- "[[C]]" if Assistant C is the best,
- ...
- "[[N]]" if Assistant N is the best,
- "[[T]]" for a tie if multiple assistants have equally good responses.

Please ensure your decision is unbiased and strictly based on the quality of the responses provided by each assistant.

"""
SINGLE_ANSWER_GRADING_PROMPT = """

[System]
Please act as an impartial judge and evaluate the quality of the responses provided by multiple AI assistants to the 
user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, 
depth, creativity, and level of detail of each response. Begin your evaluation by providing a short explanation for 
your assessment. Be as objective as possible.

[Question]
{question}

---

[Assistant A's Answer]
{answer_a}

---

[Assistant B's Answer]
{answer_b}

---

[Assistant C's Answer]
{answer_c}

...

[Assistant N's Answer]
{answer_n}

---

**Evaluation and Rating:**

To evaluate each response, consider the provided factors and rate the quality of the response on a scale of 1 to 10.

- **Assistant A:**
  - [Your evaluation and explanation for Assistant A's response]
  - Rating: [[rating_a]]

- **Assistant B:**
  - [Your evaluation and explanation for Assistant B's response]
  - Rating: [[rating_b]]

- **Assistant C:**
  - [Your evaluation and explanation for Assistant C's response]
  - Rating: [[rating_c]]

- ...

- **Assistant N:**
  - [Your evaluation and explanation for Assistant N's response]
  - Rating: [[rating_n]]

Please rate each response objectively based on its quality and how well it addresses the user's question. 
Use the scale provided (1 to 10) to assign a rating to each assistant's response. Ensure your ratings are consistent 
and reflect the merits of each response independently.

After evaluating all responses, you can determine which assistant provided the best response overall based on your 
ratings and evaluations.


"""
SINGLE_ANSWER_RATIONALE_GRADING_PROMPT = """

[System]
Please act as an impartial judge and evaluate the quality of the responses provided by multiple AI assistants to the 
user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, 
depth, creativity, and level of detail of each response. Begin your evaluation by providing a short explanation for 
your assessment. Be as objective as possible.

[Question]
{question}

---

[Assistant A's Answer]
{answer_a}

---

[Assistant B's Answer]
{answer_b}

---

[Assistant C's Answer]
{answer_c}

...

[Assistant N's Answer]
{answer_n}

---

**Evaluation and Rating:**

To evaluate each response, consider the provided factors of the response and rationales provided in the table below on 
a scale of 1 to 10.


| Rationale                           | Assistant A | Assistant B | Assistant C | ... | Assistant N |
|-------------------------------------|-------------|-------------|-------------|-----|-------------|
| Is more concise                      | [rating_a_concise] | [rating_b_concise] | [rating_c_concise] | ... | [rating_n_concise] |
| Provides clear explanations          | [rating_a_clear] | [rating_b_clear] | [rating_c_clear] | ... | [rating_n_clear] |
| Has more relevant information        | [rating_a_relevant] | [rating_b_relevant] | [rating_c_relevant] | ... | [rating_n_relevant] |
| Is more coherent and well-structured | [rating_a_coherent] | [rating_b_coherent] | [rating_c_coherent] | ... | [rating_n_coherent] |
| Has fewer factual errors             | [rating_a_accuracy] | [rating_b_accuracy] | [rating_c_accuracy] | ... | [rating_n_accuracy] |
| Avoids unsafe or biased content      | [rating_a_safe] | [rating_b_safe] | [rating_c_safe] | ... | [rating_n_safe] |
| Stays on topic                       | [rating_a_topic] | [rating_b_topic] | [rating_c_topic] | ... | [rating_n_topic] |
| Has better writing style/fluency     | [rating_a_fluency] | [rating_b_fluency] | [rating_c_fluency] | ... | [rating_n_fluency] |
| Provides more comprehensive coverage | [rating_a_comprehensive] | [rating_b_comprehensive] | [rating_c_comprehensive] | ... | [rating_n_comprehensive] |
| Is more creative/innovative          | [rating_a_creative] | [rating_b_creative] | [rating_c_creative] | ... | [rating_n_creative] |
| Gives step-by-step reasoning         | [rating_a_reasoning] | [rating_b_reasoning] | [rating_c_reasoning] | ... | [rating_n_reasoning] |
| Uses simpler language                | [rating_a_simple] | [rating_b_simple] | [rating_c_simple] | ... | [rating_n_simple] |
| Provides code/examples effectively   | [rating_a_code] | [rating_b_code] | [rating_c_code] | ... | [rating_n_code] |


Please rate each response objectively based on the rationales and how well it addresses the user's question. 
Use the scale provided (1 to 10) to assign a rating to each assistant's response for each rationale. 
Ensure your ratings are consistent and reflect the merits of each response independently.

After evaluating all responses, you can determine which assistant provided the best response overall based on your 
ratings and evaluations.
"""
REFERENCE_GUIDED_GRADING_PROMPT = """
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by multiple AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, and responses from different assistants (Assistant A, Assistant B, Assistant C, etc.). Your job is to evaluate which assistant's answer is better compared to the reference answer. Begin your evaluation by comparing each assistant's answer with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.

After providing your explanation, output your final verdict by strictly following this format:
- "[[A]]" if assistant A's answer is better,
- "[[B]]" if assistant B's answer is better,
- "[[C]]" if another assistant's answer is better,
- "[[T]]" for a tie if multiple assistants have equally good answers.

[User Question]
{question}

[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]

[Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

[Assistant C's Answer]
{answer_c}
[The End of Assistant C's Answer]

...

[Assistant N's Answer]
{answer_n}
[The End of Assistant N's Answer]

"""
