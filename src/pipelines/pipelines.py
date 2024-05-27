import src.pdf_chat_bot.utilities as utilities
from src.pdf_chat_bot.miscellanies import pdf_reader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.pdf_chat_bot import logger


class PdfToVectorPipeline:
    def __init__(self, path) -> None:
        self.path = path
        self.vector_stores = None

    def start_pipeline(self):
        logger.info("PDF to vector pipeline started for path: %s", self.path)
        try:
            texts = pdf_reader(self.path)
            logger.debug("PDF texts extracted: %s", texts)

            text_chunks = utilities.get_text_chunks(texts)
            logger.debug("Text chunks created: %d chunks", len(text_chunks))

            embeddings = utilities.get_embeddings(
                HuggingFaceEmbeddings, 'sentence-transformers/all-MiniLM-L6-v2')
            logger.debug("Embeddings model loaded: %s", embeddings)

            self.vector_stores = utilities.get_vectorstore(
                Chroma, embeddings, text_chunks)
            logger.info("PDF to vector pipeline completed successfully")

        except Exception as e:
            logger.error("Error in PDF to vector pipeline: %s", e)
            raise

    def get_vector_stores(self):
        if self.vector_stores is None:
            logger.error(
                "Vector stores have not been created yet. Run start_pipeline() first.")
            raise ValueError(
                "Vector stores have not been created yet. Run start_pipeline() first.")
        return self.vector_stores
