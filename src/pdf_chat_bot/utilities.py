from langchain.text_splitter import CharacterTextSplitter


def get_embeddings(embedding_provider, model_name: str):
    """create embeddings

    Args:
        embedding_provider: Use embedding providers like Hugging face/ Open Ai
        model_name (str): model name for embedding

    Returns:
        specific embedding instance
    """
    return embedding_provider(model_name=model_name)


def get_vectorstore(vector_store, embeddings, text_chunks: list):
    """ generate vector stores

    Args:
        text_chunks (list): chunks needed to convert to embeddings

    Returns: vector store with embedding
    """

    vector_stores = vector_store.from_texts(text_chunks, embedding=embeddings)
    return vector_stores


def get_text_chunks(text: str) -> list:
    """

    Args:
        text (str): pdf text needed to converted to chunks

    Returns:
        list: list chunks created using spitting criteria
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
