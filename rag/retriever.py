"""
Retriever Module
----------------
High-precision retriever with score filtering.
"""
def get_retriever(
    vector_store,
    k=4,
    department=None
):
    search_kwargs = {"k": k}

    if department:
        search_kwargs["filter"] = {
            "department": department
        }

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    return retriever

