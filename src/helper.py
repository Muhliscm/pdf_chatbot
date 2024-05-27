import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
import src.pdf_chat_bot.llm_chains as llm_chains
from src.pipelines.pipelines import PdfToVectorPipeline
from langchain.memory import ConversationBufferMemory
from src.pdf_chat_bot.prompt_templates import diabetes_prompt_template
from src.pdf_chat_bot import logger

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)


memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)

schema = {
    "properties": {
        "A1C": {"type": "string"}, }}


def get_diabetes_level(A1c):
    if A1c < 5.1:
        return "No Diabetes"

    elif A1c >= 5.1 and A1c <= 7:
        return "Pre-Diabetes"

    else:
        return "Diabetes"


class ConversationChain:
    def __init__(self, llm, memory, vector_stores, diabetes_prompt_template):
        self.llm = llm
        self.memory = memory
        self.vector_stores = vector_stores
        self.diabetes_prompt_template = diabetes_prompt_template
        self.conv = None
        self.logger = logger

    def create_conv_chain(self):
        self.conv = llm_chains.get_conversation_retrieval_chain(
            self.llm, self.memory, self.vector_stores, self.diabetes_prompt_template)
        self.logger.info("New conversation chain created")
        return self.conv

    def get_conv_chain(self):
        if self.conv is None:
            self.logger.error("Conversation chain has not been created yet")
            raise ValueError("Conversation chain has not been created yet")
        return self.conv

    def run_chain(self, user_inputs):
        if self.conv is None:
            self.logger.error("Conversation chain has not been created yet")
            raise ValueError("Conversation chain has not been created yet")

        self.logger.info(
            "Conversation chain run started with user inputs: %s", user_inputs)
        try:
            response = self.conv.invoke(user_inputs)
            first_llm_answer = response.get('first_llm_answer', "")
            self.logger.info(
                "Conversation chain run completed with response: %s", first_llm_answer)
            return first_llm_answer
        except Exception as e:
            self.logger.error("Error during conversation chain run: %s", e)
            raise e


class ExtractionChain:
    def __init__(self, llm, schema) -> None:
        self.llm = llm
        self.logger = logger
        self.schema = schema
        self.extract = None

    def create_extraction_chain(self):
        self.extract = llm_chains.get_extraction_chain(schema, llm)
        self.logger.info("New Extraction chain created")
        return self.extract

    def get_extraction_chain(self):
        if self.extract is None:
            self.logger.error("Extraction chain has not been created yet")
            raise ValueError("Conversation chain has not been created yet")
        return self.extract

    def run_chain(self, user_inputs):
        if self.extract is None:
            self.logger.error("Extraction chain has not been created yet")
            raise ValueError("Conversation chain has not been created yet")

        self.logger.info(
            "Extraction chain run started with user inputs: %s", user_inputs)

        try:
            responses = self.extract.invoke(user_inputs)
            text_responses = responses.get('text', [])

            if text_responses:
                results = []
                for response in text_responses:
                    a1c_value = response.get('A1C')
                    if a1c_value is not None:
                        try:
                            a1c = int(a1c_value)
                            results.append(get_diabetes_level(a1c))
                        except ValueError:
                            return "Invalid A1C value in response."
                return "".join(results) if results else "No valid A1C values found."
            else:
                return "Check your inputs"
        except Exception as e:
            logger.error("Error during extraction chain invocation: %s", e)
            return "An error occurred during extraction."


def user_input_handler():
    input_map = {1: "upload document", 2: "interact with bot", 3: "exit"}
    vector_stores = None
    conv_chain = None
    extraction_chain = None

    while True:
        user_input = input("""
        Select the options from below:
        1. Upload your document
        2. Interact with chat bot
        3. Exit

        """)

        if user_input.isnumeric() or isinstance(user_input, int):
            user_input = int(user_input)
            if user_input in input_map:
                if user_input == 1:
                    path = input("Enter path for your PDF:\n")
                    try:
                        pipeline = PdfToVectorPipeline(path)
                        pipeline.start_pipeline()
                        vector_stores = pipeline.get_vector_stores()

                        conv_chain = ConversationChain(
                            llm, memory, vector_stores, diabetes_prompt_template)
                        conv_chain.create_conv_chain()

                        extraction_chain = ExtractionChain(llm, schema)
                        extraction_chain.create_extraction_chain()

                    except Exception as e:
                        logger.error("Failed to process the PDF: %s", e)
                        continue  # Prompt the user again

                elif user_input == 2:
                    if vector_stores is None:
                        logger.warning("Please upload your document first.")
                    else:
                        user_query = input("Enter your question:\n")
                        if "A1C" in user_query.replace(" ", "").strip().upper():
                            chain = extraction_chain
                        else:
                            chain = conv_chain

                        try:
                            bot_response = chain.run_chain(
                                user_query)
                            print(f"Bot: \n{bot_response}")
                        except Exception as e:
                            logger.error(
                                "Failed to get response from bot: %s", e)
                            continue  # Prompt the user again

                elif user_input == 3:
                    print("Exiting...")
                    break
            else:
                logger.warning("Invalid input. Please select a valid option.")
        else:
            logger.warning("Invalid input. Please enter a number.")
