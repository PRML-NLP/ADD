HOTPOTQA_TASK_NAME = "HotpotQA"
HOTPOTQA_TASK_DESCRIPTION = "This is a question-answering task that includes high-quality multi-hop questions and do not contain images. It tests language modeling abilities for multi-step reasoning and covers a wide range of topics. Some questions are challenging, while others are easier, requiring multiple steps of reasoning to arrive at the final answer."

DATA_GEN_SYSTEM_PROMPT  = """I want you to be a QA pair generator to generate high-quality questions for use in Task
described as follows :
Task Name: {task_name}
Task Description: {task_description}
"""

DATA_GEN_SYSTEM_PROMPT_WITH_SEED = """I want you to be a QA pair generator to generate high-quality questions for use in Task
described as follows :
Task Name: {task_name}
Task Description: {task_description}

1. Generate a list of **10 unique topics** or fields of knowledge that are diverse and meaningful. The topics should be well-defined and not too general, ensuring that they can be used to generate specific QA pairs. Examples of fields include "Quantum Physics," "Ancient Roman History," or "Digital Marketing Strategies."

2. After generating the list, choose **one topic** randomly from the list (or let the user choose if they provide input).
   - Each question should be specific, informative, and tailored to the topic.
   - Each answer should be concise, accurate, and relevant to the question.

3. Ensure that there is no duplication in the generated QA pairs. Each question and answer should be unique across all topics.

"""

HOTPOTQA_DATA_GEN_HUMAN_PROMPT = """{QA_pairs}\nModelled on all examples above,I want you to generate new different {Gen_num} Question-Answer pairs. The format like below:
Question: The Treaty of Versailles, signed in 1919, officially ended which war?
Answer: World War I
"""


FILTERING_PROMPT = """You will be given a pair of a question and an answer. Your task is to evaluate the relationship between the question and the answer based on the following criteria:

If the question is valid and the answer correctly corresponds to the question, return True.
If the question is invalid (e.g., unclear, irrelevant, or nonsensical) or if the answer does not correctly address the question, return False.
Provide your response as either True or False based on your evaluation.
"""

SEED_GENERATION_PROMPT = """

1. Generate a list of **unique topics** or fields of knowledge that are diverse and meaningful. The topics should be well-defined and not too general, ensuring that they can be used to generate specific QA pairs. Examples of fields include "Quantum Physics," "Ancient Roman History," or "Digital Marketing Strategies."

2. Ensure that there is no duplication in the generated topics.


"""