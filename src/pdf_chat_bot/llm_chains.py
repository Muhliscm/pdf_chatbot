from langchain.chains import create_extraction_chain, RetrievalQA


def get_extraction_chain(schema, llm):
    """ llm chain to extract specific field from string

    Args:
        schema (_type_): type in output needed to return
        llm (_type_): llm instance

    Returns:
        extraction chain
    """
    return create_extraction_chain(schema=schema, llm=llm)


def get_conversation_retrieval_chain(llm, memory, vector_stores, prompt):
    """ function to create conversational chain

    Args:
        llm (_type_): llm instance
        memory (_type_): memory type
        vector_stores (_type_): vector db has vectors in it
        prompt (_type_): prompt template

    Returns:
        _type_: conversational chain for chat bot
    """

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_stores.as_retriever(search_kwargs={"k": 1}),
        memory=memory,
        chain_type_kwargs={'prompt': prompt},
        output_key="first_llm_answer"

    )
    return conversation_chain
