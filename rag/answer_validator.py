"""
Answer Validation Module
------------------------
Checks if the answer is grounded in context.
"""

def validate_answer(answer: str, context: str) -> bool:
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words) > 0.6