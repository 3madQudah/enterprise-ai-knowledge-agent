"""
Context Trimming Module
-----------------------
Keeps only lines relevant to the question.
"""

def trim_context(docs, keywords):
    filtered = []

    for doc in docs:
        for line in doc.page_content.split("\n"):
            if any(k in line.lower() for k in keywords):
                filtered.append(line)

    return "\n".join(filtered)
